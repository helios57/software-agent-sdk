"""Efficiency tools for codebase comprehension at the 1M context scale.

Tools:
- get_call_graph: Returns JSON tree of callers/callees for a symbol
- find_implementations: Lists all classes implementing an interface
- explain_dead_code: Runs static analysis to find unused declarations
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)


@dataclass
class GetCallGraphAction(Action):
    symbol_path: str
    """Format: /absolute/path/file.ext:line:char"""

    language: str = 'auto'
    """auto | python | go | typescript | java"""

    depth: int = 2
    """Max depth for the call tree."""

    @classmethod
    def name(cls) -> str:
        return 'get_call_graph'


@dataclass
class FindImplementationsAction(Action):
    interface_path: str
    """Path to the file containing the interface definition."""

    interface_name: str
    """Name of the interface/abstract class to find implementations of."""

    language: str = 'auto'

    @classmethod
    def name(cls) -> str:
        return 'find_implementations'


@dataclass
class ExplainDeadCodeAction(Action):
    directory_path: str
    """Root directory to scan for dead code."""

    languages: list[str] = field(default_factory=lambda: ['python', 'go', 'typescript'])

    @classmethod
    def name(cls) -> str:
        return 'explain_dead_code'


@dataclass
class EfficiencyObservation(Observation):
    """Structured observation for efficiency tool results."""

    result_json: str = ''
    """JSON string containing the structured result."""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EfficiencyObservation:
        text = '```json\n' + json.dumps(data, indent=2) + '\n```'
        obs = Observation.from_text(text)
        obs.__class__ = cls
        obs.result_json = json.dumps(data)
        return obs


# ── Call Graph ───────────────────────────────────────────────────────────────


class GetCallGraphTool(
    ToolDefinition[GetCallGraphAction, EfficiencyObservation]
):
    """Return the call graph for a symbol as a compact JSON tree.

    Shows everything that calls this symbol, and everything it calls.
    Gives the LLM an instant mental model of blast radius before refactoring.
    """

    name: str = 'get_call_graph'
    description: str = (
        'Return a JSON tree showing all callers and callees of a symbol. '
        'Use this BEFORE refactoring to understand blast radius. '
        'Format: /path/file.ext:line:char'
    )
    action_type: type[GetCallGraphAction] = GetCallGraphAction
    observation_type: type[EfficiencyObservation] = EfficiencyObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> GetCallGraphTool:
        return cls(executor=GetCallGraphExecutor())


class GetCallGraphExecutor:
    """Build a call graph using language-specific tools."""

    def __call__(self, action: GetCallGraphAction, conversation: Any) -> Observation:
        try:
            parts = action.symbol_path.rsplit(':', 2)
            if len(parts) != 3:
                return EfficiencyObservation.from_json(
                    {'error': 'Format: /path/file.ext:line:char'}
                )

            file_path, line, char = parts
            if not os.path.exists(file_path):
                return EfficiencyObservation.from_json(
                    {'error': f'File not found: {file_path}'}
                )

            graph = self._build_call_graph(
                file_path, int(line), int(char), action.depth
            )
            return EfficiencyObservation.from_json(graph)
        except Exception as e:
            logger.exception('get_call_graph failed')
            return EfficiencyObservation.from_json({'error': str(e)})

    def _build_call_graph(
        self, file_path: str, line: int, char: int, depth: int
    ) -> dict[str, Any]:
        """Language-agnostic call graph builder."""
        ext = os.path.splitext(file_path)[1]

        # Use ripgrep for fast call-site search (LLM-friendly)
        symbol_name = self._extract_symbol_at(file_path, line, char)
        if not symbol_name:
            return {'error': 'Could not extract symbol name'}

        callers = self._find_references(file_path, symbol_name, ext)
        callees = self._search_calls_in_func(file_path, line, ext)

        return {
            'symbol': symbol_name,
            'file': file_path,
            'line': line,
            'callers': callers[:20],  # Cap at 20 for context window
            'callees': callees[:20],
            'total_references': len(callers),
        }

    @staticmethod
    def _extract_symbol_at(file_path: str, line: int, char: int) -> str:
        """Extract the symbol name at the given position."""
        try:
            with open(file_path) as f:
                lines = f.readlines()
            if line - 1 < len(lines):
                target_line = lines[line - 1]
                # Extract word at the character position
                import re

                for m in re.finditer(r'\b([a-zA-Z_]\w*)\b', target_line):
                    if m.start() < char <= m.end():
                        return m.group(1)
                # Fallback: first word on line
                match = re.search(r'\b([a-zA-Z_]\w*)\b', target_line)
                return match.group(1) if match else ''
        except Exception:
            pass
        return ''

    @staticmethod
    def _find_references(
        file_path: str, symbol: str, ext: str
    ) -> list[dict[str, Any]]:
        """Find references using ripgrep."""
        try:
            project_root = _find_project_root(file_path)
            result = subprocess.run(
                ['rg', '--line-number', '--no-heading', r'\b' + symbol + r'\b', project_root],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_root,
            )
            refs = []
            for line in result.stdout.strip().split('\n')[:20]:
                if not line:
                    continue
                parts = line.split(':', 2)
                if len(parts) >= 2:
                    refs.append({
                        'file': parts[0],
                        'line': int(parts[1]) if parts[1].isdigit() else parts[1],
                        'snippet': parts[2][:80] if len(parts) > 2 else '',
                    })
            return refs
        except Exception:
            return []

    @staticmethod
    def _search_calls_in_func(
        file_path: str, start_line: int, ext: str
    ) -> list[str]:
        """Find function/method calls within a function body."""
        try:
            with open(file_path) as f:
                lines = f.readlines()

            # Find function scope (heuristic: next blank line or dedent)
            indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
            end_line = start_line
            for i in range(start_line, min(start_line + 200, len(lines))):
                stripped = lines[i].strip()
                if stripped and not stripped.startswith(('#', '//', '/*', '*')):
                    line_indent = len(lines[i]) - len(lines[i].lstrip())
                    if 0 < line_indent <= indent and stripped:
                        break
                    end_line = i + 1

            # Extract function/method calls (simplistic but effective)
            import re

            calls = set()
            for i in range(start_line, end_line):
                for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(', lines[i]):
                    name = m.group(1)
                    if name not in {
                        'if', 'for', 'while', 'switch', 'return', 'print',
                        'len', 'range', 'int', 'str', 'list', 'dict', 'set',
                    }:
                        calls.add(name)
            return sorted(calls)[:20]
        except Exception:
            return []


# ── Find Implementations ─────────────────────────────────────────────────────


class FindImplementationsTool(
    ToolDefinition[FindImplementationsAction, EfficiencyObservation]
):
    """Find all classes/structs implementing a given interface.

    Crucial for traversing heavily abstracted Go or Java codebases.
    """

    name: str = 'find_implementations'
    description: str = (
        'List all classes/structs that implement a given interface. '
        'Uses ripgrep to find implementations across the codebase. '
        'Essential for understanding abstract code.'
    )
    action_type: type[FindImplementationsAction] = FindImplementationsAction
    observation_type: type[EfficiencyObservation] = EfficiencyObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> FindImplementationsTool:
        return cls(executor=FindImplementationsExecutor())


class FindImplementationsExecutor:
    """Find interface implementations using ripgrep."""

    def __call__(self, action: FindImplementationsAction, conversation: Any) -> Observation:
        try:
            project_root = _find_project_root(action.interface_path)
            name = action.interface_name

            # Language-specific patterns
            patterns = {
                'go': rf'implements\b.*\b{name}\b|\)\s+\b{name}\b|^\s*\b{name}\b\s*\(',
                'java': rf'implements\s+.*\b{name}\b|extends\s+.*\b{name}\b',
                'python': rf'class\s+\w+\s*\([^)]*\b{name}\b[^)]*\)',
                'typescript': rf'implements\s+.*\b{name}\b|extends\s+.*\b{name}\b',
            }

            lang = action.language if action.language != 'auto' else _detect_lang(action.interface_path)
            pattern = patterns.get(lang, rf'\b{name}\b')

            result = subprocess.run(
                ['rg', '--line-number', '--no-heading', pattern, project_root],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_root,
            )

            implementations = []
            for line in result.stdout.strip().split('\n')[:30]:
                if not line:
                    continue
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    implementations.append({
                        'file': parts[0],
                        'line': int(parts[1]) if parts[1].isdigit() else parts[1],
                        'snippet': parts[2].strip()[:100],
                    })

            return EfficiencyObservation.from_json({
                'interface': name,
                'file': action.interface_path,
                'implementations': implementations,
                'count': len(implementations),
            })
        except Exception as e:
            return EfficiencyObservation.from_json({'error': str(e)})


# ── Dead Code Detection ──────────────────────────────────────────────────────


class ExplainDeadCodeTool(
    ToolDefinition[ExplainDeadCodeAction, EfficiencyObservation]
):
    """Find unused declarations that can be safely removed.

    Runs language-specific static analysis to identify unused exports,
    private methods, or dead imports. Helps keep the codebase lean.
    """

    name: str = 'explain_dead_code'
    description: str = (
        'Run static analysis to identify unused exports, private methods, '
        'and dead imports in the given directory. Helps keep the context '
        'window lean for future turns.'
    )
    action_type: type[ExplainDeadCodeAction] = ExplainDeadCodeAction
    observation_type: type[EfficiencyObservation] = EfficiencyObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> ExplainDeadCodeTool:
        return cls(executor=ExplainDeadCodeExecutor())


class ExplainDeadCodeExecutor:
    """Detect dead code using language-specific tools."""

    def __call__(self, action: ExplainDeadCodeAction, conversation: Any) -> Observation:
        results: dict[str, Any] = {}
        try:
            for lang in action.languages:
                results[lang] = self._detect_for_lang(action.directory_path, lang)
            return EfficiencyObservation.from_json(results)
        except Exception as e:
            return EfficiencyObservation.from_json({'error': str(e)})

    @staticmethod
    def _detect_for_lang(directory: str, lang: str) -> dict[str, Any]:
        try:
            if lang == 'python':
                return _detect_python_dead(directory)
            if lang == 'typescript':
                return _detect_ts_dead(directory)
            if lang == 'go':
                return _detect_go_dead(directory)
            return {'unsupported': f'Dead code detection for {lang} not yet implemented'}
        except Exception as e:
            return {'error': str(e)}


def _detect_python_dead(directory: str) -> dict[str, Any]:
    """Use vulture for Python dead code detection."""
    try:
        result = subprocess.run(
            ['vulture', directory, '--min-confidence', '70'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        dead_items = []
        for line in result.stdout.strip().split('\n')[:50]:
            if line.strip():
                dead_items.append(line.strip())
        return {'tool': 'vulture', 'dead_items': dead_items, 'count': len(dead_items)}
    except Exception:
        return {'tool': 'vulture', 'error': 'vulture not available'}


def _detect_ts_dead(directory: str) -> dict[str, Any]:
    """Use ts-prune for TypeScript dead code detection."""
    try:
        result = subprocess.run(
            ['npx', 'ts-prune'],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=directory,
        )
        dead_items = []
        for line in result.stdout.strip().split('\n')[:50]:
            if line.strip():
                dead_items.append(line.strip())
        return {'tool': 'ts-prune', 'dead_items': dead_items, 'count': len(dead_items)}
    except Exception:
        return {'tool': 'ts-prune', 'error': 'ts-prune not available'}


def _detect_go_dead(directory: str) -> dict[str, Any]:
    """Use deadcode for Go dead code detection."""
    try:
        result = subprocess.run(
            ['deadcode', './...'],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=directory,
        )
        dead_items = []
        for line in result.stdout.strip().split('\n')[:50]:
            if line.strip():
                dead_items.append(line.strip())
        return {'tool': 'deadcode', 'dead_items': dead_items, 'count': len(dead_items)}
    except Exception:
        return {'tool': 'deadcode', 'error': 'deadcode not available'}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_project_root(path: str) -> str:
    """Walk up to find .git or go.mod or package.json."""
    current = os.path.abspath(os.path.dirname(path))
    markers = ['.git', 'go.mod', 'package.json', 'pyproject.toml', 'Cargo.toml']
    while current != '/':
        for marker in markers:
            if os.path.exists(os.path.join(current, marker)):
                return current
        current = os.path.dirname(current)
    return os.path.abspath(os.path.dirname(path))


def _detect_lang(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1]
    return {
        '.py': 'python',
        '.go': 'go',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
    }.get(ext, 'unknown')


__all__ = [
    'GetCallGraphAction',
    'FindImplementationsAction',
    'ExplainDeadCodeAction',
    'GetCallGraphTool',
    'FindImplementationsTool',
    'ExplainDeadCodeTool',
]
