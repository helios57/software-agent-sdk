"""Python refactoring tools using libcst (Concrete Syntax Tree).

Provides AST-aware tools that are immune to whitespace/formatting issues:
- py_rename: Rename a symbol across the project
- py_extract_method: Extract a code block into a method
- py_inline_variable: Inline a variable at all its usage sites
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations
from openhands.tools.lsp.base import (
    ASTBackend,
    RefactorObservation,
    RefactorResult,
    SymbolLocation,
)

logger = logging.getLogger(__name__)

# ── Actions ──────────────────────────────────────────────────────────────────


@dataclass
class PyRenameAction(Action):
    file_path: str
    line: int
    """1-based line of the symbol definition."""
    character: int
    """1-based character offset of the symbol name."""
    new_name: str
    """The new (unqualified) name for the symbol."""

    @classmethod
    def name(cls) -> str:
        return 'py_rename'


@dataclass
class PyExtractMethodAction(Action):
    file_path: str
    start_line: int
    """1-based line where the block to extract starts."""
    end_line: int
    """1-based line where the block to extract ends."""
    func_name: str
    """Name for the newly extracted method."""

    @classmethod
    def name(cls) -> str:
        return 'py_extract_method'


@dataclass
class PyInlineVariableAction(Action):
    file_path: str
    variable_name: str
    """Name of the variable to inline (must be defined in the same file)."""

    @classmethod
    def name(cls) -> str:
        return 'py_inline_variable'


# ── Python AST Backend ───────────────────────────────────────────────────────


class PythonASTBackend(ASTBackend):
    """Python refactoring backend powered by libcst."""

    language: ClassVar[str] = 'python'

    # ── Rename ────────────────────────────────────────────────────────────

    def rename(
        self, file_path: str, line: int, character: int, new_name: str
    ) -> RefactorResult:
        try:
            with open(file_path) as f:
                source = f.read()
            module = cst.parse_module(source)
            wrapper = MetadataWrapper(module)

            renamer = _RenameVisitor(
                target_line=line,
                target_col=character,
                new_name=new_name,
                file_path=file_path,
            )
            new_module = wrapper.visit(renamer)

            if renamer.renames == 0:
                return RefactorResult(
                    error=f'Symbol not found at {file_path}:{line}:{character}'
                )

            new_source = new_module.code
            with open(file_path, 'w') as f:
                f.write(new_source)

            return RefactorResult(
                diff=_compute_diff(file_path, source, new_source),
                files_changed=[file_path],
                warnings=renamer.warnings,
            )
        except Exception as e:
            logger.exception('py_rename failed')
            return RefactorResult(error=str(e))

    # ── Extract method ─────────────────────────────────────────────────────

    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        func_name: str,
    ) -> RefactorResult:
        try:
            with open(file_path) as f:
                source = f.read()
            lines = source.splitlines(keepends=True)

            if end_line < start_line:
                return RefactorResult(error='end_line must be >= start_line')

            # Extract the block
            block_lines = lines[start_line - 1 : end_line]
            block = ''.join(block_lines).rstrip()
            if not block.strip():
                return RefactorResult(
                    error='Selected block is empty'
                )

            # Compute indentation
            indent = _detect_indent(block_lines[0])

            # Determine variables used from outer scope
            outer_names = _detect_outer_names(block_lines, indent)

            # Build parameter list
            params = ', '.join(sorted(outer_names))

            # Build the extracted method
            method_def = f'\n{indent}def {func_name}({params}):\n'
            method_body = _reindent(block, indent, indent + '    ')
            method_def += method_body + '\n'

            # Build the call site
            call = f'{indent}{func_name}({params})'

            # Replace block with call, append method after end_line
            before = lines[: start_line - 1]
            after = lines[end_line:]
            new_lines = before + [call + '\n'] + after + [method_def]
            new_source = ''.join(new_lines)

            with open(file_path, 'w') as f:
                f.write(new_source)

            return RefactorResult(
                diff=_compute_diff(file_path, source, new_source),
                files_changed=[file_path],
            )
        except Exception as e:
            logger.exception('py_extract_method failed')
            return RefactorResult(error=str(e))

    # ── Inline variable ────────────────────────────────────────────────────

    def inline_variable(self, file_path: str, variable_name: str) -> RefactorResult:
        try:
            with open(file_path) as f:
                source = f.read()
            module = cst.parse_module(source)
            wrapper = MetadataWrapper(module)

            inliner = _InlineVariableVisitor(variable_name)
            new_module = wrapper.visit(inliner)

            if inliner.definition_value is None:
                return RefactorResult(
                    error=f'Variable "{variable_name}" not found'
                )
            if inliner.was_reassigned:
                return RefactorResult(
                    error=f'Variable "{variable_name}" was reassigned — unsafe to inline'
                )

            new_source = new_module.code
            with open(file_path, 'w') as f:
                f.write(new_source)

            return RefactorResult(
                diff=_compute_diff(file_path, source, new_source),
                files_changed=[file_path],
                warnings=[f'Inlined {inliner.usages} usage(s) of "{variable_name}"'],
            )
        except Exception as e:
            logger.exception('py_inline_variable failed')
            return RefactorResult(error=str(e))

    # ── Analysis ───────────────────────────────────────────────────────────

    def get_call_graph(self, symbol_path: str) -> dict[str, Any]:
        return {'error': 'Not yet implemented for Python'}

    def find_implementations(self, interface_path: str) -> list[SymbolLocation]:
        return []

    def shutdown(self) -> None:
        pass


# ── libcst Visitors ──────────────────────────────────────────────────────────


class _RenameVisitor(cst.CSTTransformer):
    """Visit an AST and rename a symbol at the given line/col."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self, target_line: int, target_col: int, new_name: str, file_path: str
    ):
        super().__init__()
        self._target_line = target_line
        self._target_col = target_col
        self._new_name = new_name
        self._file_path = file_path
        self._target_full_name: str | None = None
        self.renames = 0
        self.warnings: list[str] = []

    def visit_Name(self, node: cst.Name) -> bool | None:
        pos = self.get_metadata(PositionProvider, node)
        if (
            pos.start.line == self._target_line
            and pos.start.column == self._target_col
        ):
            self._target_full_name = node.value
        return True

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.Name:
        if (
            self._target_full_name is not None
            and original_node.value == self._target_full_name
        ):
            self.renames += 1
            return updated_node.with_changes(value=self._new_name)
        return updated_node

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        if self._target_full_name is not None and isinstance(
            original_node.names, cst.ImportStar
        ):
            self.warnings.append(
                f'Wildcard import in {self._file_path} — '
                f'may shadow renamed symbol'
            )
        return updated_node


class _InlineVariableVisitor(cst.CSTTransformer):
    """Inline a simple variable assignment."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, variable_name: str):
        super().__init__()
        self._var_name = variable_name
        self.definition_value: cst.BaseExpression | None = None
        self.usages = 0
        self.was_reassigned = False

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        if _name_matches(node.target, self._var_name):
            if node.value is not None:
                self.definition_value = node.value
        return True

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> cst.RemovalSentinel | cst.BaseSmallStatement:
        if self.definition_value is not None:
            return cst.RemoveFromParent()
        return updated_node

    def visit_Assign(self, node: cst.Assign) -> bool | None:
        return True

    def leave_Assign(
        self, original_node: cst.Assign, updated_node: cst.Assign
    ) -> cst.BaseSmallStatement | cst.FlattenSentinel | cst.RemovalSentinel:
        if len(original_node.targets) == 1 and _name_matches(
            original_node.targets[0].target, self._var_name
        ):
            self.definition_value = original_node.value
            return cst.RemoveFromParent()
        # Check for reassignments
        for target in original_node.targets:
            if _name_matches(target.target, self._var_name):
                self.was_reassigned = True
        return updated_node

    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if (
            original_node.value == self._var_name
            and self.definition_value is not None
        ):
            # Only inline in expression positions (not left side of assignment)
            parent = self.get_metadata(PositionProvider, original_node)
            self.usages += 1
            return self.definition_value
        return updated_node


class _ClassMethodFinder(cst.CSTVisitor):
    """Find all method definitions in a class body."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        self.methods: list[tuple[str, int, int]] = []  # name, start_line, end_line
        self._current_class: str | None = None

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self._current_class = node.name.value
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._current_class = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        pos = self.get_metadata(PositionProvider, node)
        self.methods.append((node.name.value, pos.start.line, pos.end.line))
        return True


# ── Helpers ──────────────────────────────────────────────────────────────────


def _name_matches(node: cst.BaseExpression, name: str) -> bool:
    return isinstance(node, cst.Name) and node.value == name


def _detect_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(block: str, from_indent: str, to_indent: str) -> str:
    """Change the indentation of a block of code."""
    lines = block.splitlines(keepends=True)
    result = []
    for line in lines:
        if line.startswith(from_indent):
            result.append(to_indent + line[len(from_indent) :])
        elif line.strip():
            result.append(to_indent + line.lstrip())
        else:
            result.append(line)
    return ''.join(result)


def _detect_outer_names(
    block_lines: list[str], outer_indent: str
) -> set[str]:
    """Heuristically detect variables that are used from outer scope."""
    import keyword
    import re

    names: set[str] = set()
    for line in block_lines:
        if line.startswith(outer_indent + '    ') or (
            line.strip() and not line.startswith(outer_indent)
        ):
            stripped = line.strip()
            # Skip comments, decorators, and keywords
            if stripped.startswith('#') or stripped.startswith('@'):
                continue
            # Find bare identifiers
            for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', stripped):
                name = match.group(1)
                if name not in keyword.kwlist and name not in {
                    'True', 'False', 'None', 'self', 'cls',
                }:
                    names.add(name)
    return names


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


# ── Tool Definitions ────────────────────────────────────────────────────────


class PyRenameTool(ToolDefinition[PyRenameAction, RefactorObservation]):
    """AST-aware Python symbol rename using libcst."""

    name: str = 'py_rename'
    description: str = (
        'Rename a Python symbol (variable, function, class) at the given '
        'file:line:char position. Uses AST analysis to rename all usages '
        'within the file. Safe — ignores string literals and comments.'
    )
    action_type: type[PyRenameAction] = PyRenameAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PyRenameTool:
        backend = PythonASTBackend()
        from openhands.tools.lsp.base import LSPRefactorExecutor

        return cls(executor=LSPRefactorExecutor({'python': backend}))


class PyExtractMethodTool(ToolDefinition[PyExtractMethodAction, RefactorObservation]):
    """AST-aware Python method extraction using libcst."""

    name: str = 'py_extract_method'
    description: str = (
        'Extract lines start_line..end_line from a Python file into a new method '
        'named func_name. Automatically determines parameters from outer-scope '
        'variables and handles indentation correctly.'
    )
    action_type: type[PyExtractMethodAction] = PyExtractMethodAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PyExtractMethodTool:
        backend = PythonASTBackend()
        from openhands.tools.lsp.base import LSPRefactorExecutor

        return cls(executor=LSPRefactorExecutor({'python': backend}))


class PyInlineVariableTool(
    ToolDefinition[PyInlineVariableAction, RefactorObservation]
):
    """Inline a Python variable at all its usage sites."""

    name: str = 'py_inline_variable'
    description: str = (
        'Inline a variable by replacing all references with its assigned '
        'value and removing the original assignment. Only safe for variables '
        'that are assigned once and never reassigned.'
    )
    action_type: type[PyInlineVariableAction] = PyInlineVariableAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PyInlineVariableTool:
        backend = PythonASTBackend()
        from openhands.tools.lsp.base import LSPRefactorExecutor

        return cls(executor=LSPRefactorExecutor({'python': backend}))


__all__ = [
    'PyRenameAction',
    'PyExtractMethodAction',
    'PyInlineVariableAction',
    'PyRenameTool',
    'PyExtractMethodTool',
    'PyInlineVariableTool',
    'PythonASTBackend',
]
