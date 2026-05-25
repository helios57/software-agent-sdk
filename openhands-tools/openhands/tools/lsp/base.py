"""LSP/AST-based refactoring tools — foundation layer.

Architecture: ToolDefinition → Executor → Backend (LSP server or AST library).

Each tool:
1. Accepts a structured Action with file/line/char targeting
2. Routes to the appropriate language-specific backend
3. Returns an Observation with a diff and TypeCheck status
"""

from __future__ import annotations

import abc
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, ClassVar

from openhands.sdk.tool.schema import Action, Observation, Schema
from openhands.sdk.tool.tool import ToolDefinition, ToolExecutor

logger = logging.getLogger(__name__)


# ── Schemas ──────────────────────────────────────────────────────────────────


class RefactorRequest(Schema):
    """Base for all refactoring actions."""

    file_path: str
    """Absolute path to the target file."""


class RefactorResult(Schema):
    """Structured result from a refactoring operation."""

    diff: str = ''
    """Unified diff of all changes made across files."""

    files_changed: list[str] = field(default_factory=list)
    """Absolute paths of files that were modified."""

    type_check_passed: bool | None = None
    """Whether background type-checking passed (None if not applicable)."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings (e.g., about edge cases)."""

    error: str | None = None
    """Fatal error message if the operation failed."""


class RefactorObservation(Observation):
    """Observation returned by all LSP/AST refactoring tools."""

    result: RefactorResult = field(default_factory=RefactorResult)

    @classmethod
    def from_result(cls, result: RefactorResult) -> RefactorObservation:
        text_parts: list[str] = []
        if result.error:
            text_parts.append(f'❌ REFACTOR FAILED: {result.error}')
        else:
            text_parts.append(
                f'✅ Refactored {len(result.files_changed)} file(s)'
            )
            if result.type_check_passed is True:
                text_parts.append('TypeCheck: PASS')
            elif result.type_check_passed is False:
                text_parts.append('TypeCheck: FAIL')
            if result.warnings:
                text_parts.append('Warnings:')
                text_parts.extend(f'  ⚠ {w}' for w in result.warnings)

        if result.diff:
            text_parts.append(f'\n```diff\n{result.diff}\n```')

        obs = Observation.from_text('\n'.join(text_parts))
        obs.is_error = result.error is not None
        obs.__class__ = cls  # type: ignore[assignment]
        obs.result = result  # type: ignore[attr-defined]
        return obs  # type: ignore[return-value]


# ── Backend Interface ────────────────────────────────────────────────────────


@dataclass
class SymbolLocation:
    """A precise location in source code (LSP-compatible)."""

    uri: str
    """file:// URI of the containing file."""

    line: int
    """1-based line number."""

    character: int
    """1-based character offset on the line."""

    end_line: int | None = None
    end_character: int | None = None


class ASTBackend(abc.ABC):
    """Abstract interface for language-specific AST/LSP backends.

    Each backend is responsible for:
    - Starting/stopping any language server processes
    - Translating tool actions into language-specific AST mutations
    - Returning unified diffs + type-check results
    """

    language: ClassVar[str]
    """Identifier: 'python', 'go', 'typescript', 'java', 'proto'."""

    @abc.abstractmethod
    def rename(self, file_path: str, line: int, character: int, new_name: str) -> RefactorResult: ...

    @abc.abstractmethod
    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        func_name: str,
    ) -> RefactorResult: ...

    @abc.abstractmethod
    def get_call_graph(self, symbol_path: str) -> dict[str, Any]: ...

    @abc.abstractmethod
    def find_implementations(self, interface_path: str) -> list[SymbolLocation]: ...

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up any language server processes."""


# ── LSP Server Manager ───────────────────────────────────────────────────────


class LSPServer:
    """Manages a single Language Server Protocol process.

    Handles JSON-RPC lifecycle: initialize → didOpen → request → shutdown → exit.
    """

    _lsp_logger: ClassVar = logging.getLogger('openhands.tools.lsp.jsonrpc')

    def __init__(self, command: list[str], workspace_root: str):
        self._command = command
        self._workspace_root = workspace_root
        self._proc: subprocess.Popen | None = None
        self._seq = 0
        self._initialized = False

    def start(self) -> None:
        """Launch the LSP server process and send initialize."""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._workspace_root,
        )
        self._send_request('initialize', {
            'processId': os.getpid(),
            'rootUri': f'file://{self._workspace_root}',
            'capabilities': {
                'textDocument': {
                    'rename': {'dynamicRegistration': True},
                    'documentSymbol': {},
                    'references': {},
                    'implementation': {},
                    'callHierarchy': {'dynamicRegistration': True},
                },
                'workspace': {
                    'applyEdit': True,
                    'workspaceEdit': {'documentChanges': True},
                },
            },
        })
        self._send_notification('initialized', {})
        self._initialized = True
        self._lsp_logger.debug('LSP server initialized: %s', self._command[0])

    def shutdown(self) -> None:
        """Gracefully shut down the LSP server."""
        if self._proc is None:
            return
        try:
            self._send_request('shutdown', {})
            self._send_notification('exit', {})
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        finally:
            self._proc = None
            self._initialized = False

    def open_document(self, file_path: str, text: str) -> None:
        """Register a document with the LSP server so it can be analysed."""
        uri = f'file://{file_path}'
        self._send_notification('textDocument/didOpen', {
            'textDocument': {
                'uri': uri,
                'languageId': self._detect_language(file_path),
                'version': 1,
                'text': text,
            },
        })

    def request_rename(
        self, file_path: str, line: int, character: int, new_name: str
    ) -> dict[str, Any] | None:
        """Request a workspace/rename from the LSP server."""
        return self._send_request('textDocument/rename', {
            'textDocument': {'uri': f'file://{file_path}'},
            'position': {'line': line - 1, 'character': character - 1},
            'newName': new_name,
        })

    def request_references(
        self, file_path: str, line: int, character: int
    ) -> list[dict[str, Any]]:
        """Find all references to a symbol."""
        return self._send_request('textDocument/references', {
            'textDocument': {'uri': f'file://{file_path}'},
            'position': {'line': line - 1, 'character': character - 1},
            'context': {'includeDeclaration': True},
        }) or []

    def request_implementations(
        self, file_path: str, line: int, character: int
    ) -> list[dict[str, Any]]:
        """Find all implementations of an interface."""
        return self._send_request('textDocument/implementation', {
            'textDocument': {'uri': f'file://{file_path}'},
            'position': {'line': line - 1, 'character': character - 1},
        }) or []

    def request_call_hierarchy_incoming(
        self, file_path: str, line: int, character: int
    ) -> list[dict[str, Any]]:
        """Find all callers of a function via call hierarchy."""
        item = self._send_request('textDocument/prepareCallHierarchy', {
            'textDocument': {'uri': f'file://{file_path}'},
            'position': {'line': line - 1, 'character': character - 1},
        })
        if not item:
            return []
        return self._send_request('callHierarchy/incomingCalls', {
            'item': item[0] if isinstance(item, list) else item,
        }) or []

    # ── JSON-RPC ──────────────────────────────────────────────────────────

    def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        self._seq += 1
        msg = json.dumps({
            'jsonrpc': '2.0',
            'id': self._seq,
            'method': method,
            'params': params,
        })
        self._write(msg)
        return self._read_response(self._seq)

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        msg = json.dumps({
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
        })
        self._write(msg)

    def _write(self, content: str) -> None:
        assert self._proc and self._proc.stdin
        header = f'Content-Length: {len(content)}\r\n\r\n'
        self._proc.stdin.write(header.encode() + content.encode())
        self._proc.stdin.flush()

    def _read_response(self, expected_id: int) -> Any:
        while True:
            header = b''
            while not header.endswith(b'\r\n\r\n'):
                chunk = (
                    self._proc.stdout.read(1)
                    if self._proc and self._proc.stdout
                    else b''
                )
                if not chunk:
                    return None
                header += chunk
            length = int(
                header.split(b'Content-Length: ')[1].split(b'\r\n')[0]
            )
            body = (
                self._proc.stdout.read(length)
                if self._proc and self._proc.stdout
                else b''
            )
            msg = json.loads(body)
            if msg.get('id') == expected_id:
                if 'error' in msg:
                    self._lsp_logger.warning(
                        'LSP error: %s',
                        msg['error'].get('message', str(msg['error'])),
                    )
                    return None
                return msg.get('result')

    @staticmethod
    def _detect_language(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1]
        return {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'typescriptreact',
            '.go': 'go',
            '.java': 'java',
            '.proto': 'proto',
        }.get(ext, 'plaintext')


# ── LSPBackend ───────────────────────────────────────────────────────────────


class LSPBackend(ASTBackend):
    """Generic backend powered by a Language Server Protocol server."""

    def __init__(self, server: LSPServer):
        self._server = server
        self._server.start()

    def rename(
        self, file_path: str, line: int, character: int, new_name: str
    ) -> RefactorResult:
        with open(file_path) as f:
            text = f.read()
        self._server.open_document(file_path, text)

        result = self._server.request_rename(file_path, line, character, new_name)
        if result is None:
            return RefactorResult(error='Rename request failed — symbol not found')

        return _apply_workspace_edit(
            result.get('changes', {}),
            result.get('documentChanges', []),
        )

    def extract_function(
        self, file_path: str, start_line: int, end_line: int, func_name: str
    ) -> RefactorResult:
        raise NotImplementedError(
            f'extract_function not supported via LSP for {self.language}'
        )

    def get_call_graph(self, symbol_path: str) -> dict[str, Any]:
        return {}

    def find_implementations(self, interface_path: str) -> list[SymbolLocation]:
        with open(interface_path) as f:
            text = f.read()
        self._server.open_document(interface_path, text)

        results = self._server.request_implementations(interface_path, 1, 1)
        if not results:
            return []
        return [
            SymbolLocation(
                uri=loc.get('uri', ''),
                line=loc.get('range', {}).get('start', {}).get('line', 0) + 1,
                character=loc.get('range', {}).get('start', {}).get('character', 0) + 1,
            )
            for loc in results
        ]

    def shutdown(self) -> None:
        self._server.shutdown()


# ── Workspace edit applicator ────────────────────────────────────────────────


def _apply_workspace_edit(
    changes: dict[str, list[dict[str, Any]]],
    document_changes: list[dict[str, Any]],
) -> RefactorResult:
    """Apply text edits from a workspace/rename response and compute a diff."""
    edits_by_file: dict[str, list[dict[str, Any]]] = {}

    for uri, text_edits in changes.items():
        file_path = uri.replace('file://', '')
        edits_by_file.setdefault(file_path, []).extend(text_edits)

    for doc_change in document_changes:
        if 'textDocument' in doc_change:
            uri = doc_change['textDocument']['uri']
            file_path = uri.replace('file://', '')
            edits_by_file.setdefault(file_path, []).extend(
                doc_change.get('edits', [])
            )

    if not edits_by_file:
        return RefactorResult(error='No edits to apply')

    diff_parts: list[str] = []
    files_changed: list[str] = []

    for file_path, edits in edits_by_file.items():
        if not os.path.exists(file_path):
            continue
        with open(file_path) as f:
            original = f.read()

        modified = original
        for edit in sorted(edits, key=_edit_sort_key, reverse=True):
            start = _position_to_offset(modified, edit['range']['start'])
            end = _position_to_offset(modified, edit['range']['end'])
            modified = modified[:start] + edit['newText'] + modified[end:]

        if modified != original:
            with open(file_path, 'w') as f:
                f.write(modified)
            files_changed.append(file_path)
            diff_parts.append(_compute_diff(file_path, original, modified))

    return RefactorResult(
        diff='\n'.join(diff_parts),
        files_changed=files_changed,
    )


def _edit_sort_key(edit: dict[str, Any]) -> int:
    start = edit['range']['start']
    return start['line'] * 100000 + start['character']


def _position_to_offset(text: str, position: dict[str, int]) -> int:
    lines = text.splitlines(keepends=True)
    offset = 0
    for i in range(position['line']):
        if i < len(lines):
            offset += len(lines[i])
    if position['line'] < len(lines):
        offset += min(position['character'], len(lines[position['line']]))
    return offset


def _compute_diff(file_path: str, original: str, modified: str) -> str:
    import difflib

    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    return ''.join(diff_lines)


# ── Universal Executor ───────────────────────────────────────────────────────


class LSPRefactorExecutor(ToolExecutor):
    """Executor that dispatches refactoring actions to the appropriate backend."""

    def __init__(self, backends: dict[str, ASTBackend]):
        self._backends = backends

    def __call__(self, action: Action, conversation: Any) -> Observation:
        if not hasattr(action, 'file_path'):
            return Observation.from_text(
                'Action lacks file_path', is_error=True
            )

        backend = self._backend_for(action.file_path)
        if backend is None:
            return Observation.from_text(
                f'No refactoring backend available for {action.file_path}',
                is_error=True,
            )

        result = self._dispatch(action, backend)
        return RefactorObservation.from_result(result)

    def _backend_for(self, file_path: str) -> ASTBackend | None:
        ext = os.path.splitext(file_path)[1]
        lang = {
            '.py': 'python',
            '.go': 'go',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.js': 'typescript',
            '.jsx': 'typescript',
            '.java': 'java',
            '.proto': 'proto',
        }.get(ext)
        return self._backends.get(lang) if lang else None

    def _dispatch(self, action: Any, backend: ASTBackend) -> RefactorResult:
        action_name = (
            getattr(type(action), 'name', None) or type(action).__name__
        )

        if action_name.endswith('_rename'):
            return backend.rename(
                action.file_path, action.line, action.character, action.new_name
            )

        if action_name.endswith('_extract_method') or action_name.endswith(
            '_extract_function'
        ):
            return backend.extract_function(
                action.file_path,
                action.start_line,
                action.end_line,
                action.func_name,
            )

        if action_name == 'get_call_graph':
            return backend.get_call_graph(
                getattr(action, 'symbol_path', action.file_path)
            )

        if action_name == 'find_implementations':
            return backend.find_implementations(action.file_path)

        raise ValueError(f'Unknown refactoring action: {action_name}')


__all__ = [
    'ASTBackend',
    'LSPBackend',
    'LSPServer',
    'LSPRefactorExecutor',
    'RefactorRequest',
    'RefactorResult',
    'RefactorObservation',
    'SymbolLocation',
]
