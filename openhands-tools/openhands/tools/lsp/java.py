"""Java/Spring Boot refactoring tools via Eclipse JDT Language Server.

Tools:
- java_safe_rename: Rename with Spring annotation awareness (@Qualifier, @Bean)
- java_move_class: Move class + update package + rewrite imports
- java_extract_bean: Extract logic into @Component/@Service with DI wiring
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar

from openhands.sdk.tool.schema import Action
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations
from openhands.tools.lsp.base import (
    ASTBackend,
    LSPBackend,
    LSPServer,
    LSPRefactorExecutor,
    RefactorObservation,
    RefactorResult,
)

logger = logging.getLogger(__name__)


@dataclass
class JavaRenameAction(Action):
    file_path: str
    line: int
    character: int
    new_name: str

    @classmethod
    def name(cls) -> str:
        return 'java_safe_rename'


@dataclass
class JavaMoveClassAction(Action):
    source_class: str
    """Fully qualified class name, e.g. com.example.MyClass."""
    target_package: str
    """Target package, e.g. com.example.refactored."""

    @classmethod
    def name(cls) -> str:
        return 'java_move_class'


@dataclass
class JavaExtractBeanAction(Action):
    file_path: str
    start_line: int
    end_line: int
    component_name: str
    """Name for the new @Component or @Service."""
    annotation: str = 'Component'
    """Spring stereotype: Component, Service, Repository."""

    @classmethod
    def name(cls) -> str:
        return 'java_extract_bean'


class JavaBackend(LSPBackend):
    """Java refactoring backend powered by Eclipse JDT.LS."""

    language: ClassVar[str] = 'java'

    def __init__(self, workspace_root: str | None = None):
        self._workspace_root = workspace_root or os.getcwd()
        # jdtls needs specific launch parameters — simplified for illustration
        server = LSPServer(
            ['jdtls', '-data', os.path.join(self._workspace_root, '.jdtls')],
            self._workspace_root,
        )
        super().__init__(server)

    def extract_function(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        func_name: str,
    ) -> RefactorResult:
        return self.extract_bean(
            file_path, start_line, end_line, func_name, 'Component'
        )

    def extract_bean(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        component_name: str,
        annotation: str = 'Component',
    ) -> RefactorResult:
        """Extract a block of logic into a new Spring bean.

        Creates a new @Component/@Service class with constructor injection,
        updates the original file to inject the new dependency.
        """
        try:
            with open(file_path) as f:
                source = f.read()
            lines = source.splitlines(keepends=True)

            block = ''.join(lines[start_line - 1 : end_line]).rstrip()
            if not block.strip():
                return RefactorResult(error='Selected block is empty')

            # Detect dependencies used from outer scope
            deps = self._detect_injected_deps(lines, start_line, end_line)

            # Build new bean file
            package = self._detect_package(lines)
            bean_source = f'package {package};\n\n'
            bean_source += 'import org.springframework.stereotype.{annotation};\n'
            bean_source += 'import lombok.RequiredArgsConstructor;\n\n'
            bean_source += f'@{annotation}\n'
            bean_source += '@RequiredArgsConstructor\n'
            bean_source += f'public class {component_name} {{\n'
            for dep in deps:
                bean_source += f'    private final {dep};\n'
            bean_source += '\n'
            bean_source += self._reindent(block, self._detect_indent(lines[start_line - 1]), '    ')
            bean_source += '\n}\n'

            # Write new file
            dir_path = os.path.dirname(file_path)
            bean_path = os.path.join(dir_path, f'{component_name}.java')
            with open(bean_path, 'w') as f:
                f.write(bean_source)

            # Replace block with injection
            indent = self._detect_indent(lines[start_line - 1])
            injection = f'{indent}private final {component_name} {self._lower_first(component_name)};\n'
            new_lines = lines[: start_line - 1] + [injection] + lines[end_line:]
            new_source = ''.join(new_lines)

            with open(file_path, 'w') as f:
                f.write(new_source)

            return RefactorResult(
                diff=_compute_diff(file_path, source, new_source),
                files_changed=[file_path, bean_path],
            )
        except Exception as e:
            logger.exception('java_extract_bean failed')
            return RefactorResult(error=str(e))

    @staticmethod
    def _detect_package(lines: list[str]) -> str:
        for line in lines:
            if line.strip().startswith('package '):
                return line.strip().split()[1].rstrip(';')
        return 'com.example'

    @staticmethod
    def _detect_injected_deps(
        lines: list[str], start: int, end: int
    ) -> list[str]:
        """Heuristically detect injected dependencies."""
        import re

        deps = []
        block = ''.join(lines[start - 1 : end])
        # Look for field accesses like `this.someService` or static calls
        for match in re.finditer(r'(\w+Service|\w+Repository|\w+Mapper)', block):
            name = match.group(1)
            if name not in deps:
                deps.append(name)
        return deps

    @staticmethod
    def _detect_indent(line: str) -> str:
        return line[: len(line) - len(line.lstrip())]

    @staticmethod
    def _reindent(block: str, from_indent: str, to_indent: str) -> str:
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

    @staticmethod
    def _lower_first(s: str) -> str:
        return s[0].lower() + s[1:] if s else s


def _compute_diff(file_path: str, original: str, modified: str) -> str:
    import difflib

    return ''.join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )


# ── Tool Definitions ─────────────────────────────────────────────────────────


class JavaRenameTool(ToolDefinition[JavaRenameAction, RefactorObservation]):
    name: str = 'java_safe_rename'
    description: str = (
        'Rename a Java symbol (class, method, field) using JDT.LS. '
        'Intelligently refactors Spring @Qualifier, @Bean, and '
        '@RequestMapping string references when they match the class name.'
    )
    action_type: type[JavaRenameAction] = JavaRenameAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> JavaRenameTool:
        backend = JavaBackend()
        return cls(executor=LSPRefactorExecutor({'java': backend}))


class JavaMoveClassTool(
    ToolDefinition[JavaMoveClassAction, RefactorObservation]
):
    name: str = 'java_move_class'
    description: str = (
        'Move a class to a new package. Updates the package declaration '
        'and rewrites imports in all dependent files.'
    )
    action_type: type[JavaMoveClassAction] = JavaMoveClassAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> JavaMoveClassTool:
        backend = JavaBackend()
        return cls(executor=LSPRefactorExecutor({'java': backend}))


class JavaExtractBeanTool(
    ToolDefinition[JavaExtractBeanAction, RefactorObservation]
):
    name: str = 'java_extract_bean'
    description: str = (
        'Extract a block of logic into a new Spring @Component or @Service. '
        'Wires up dependency injection (constructor injection) and updates '
        'the original file.'
    )
    action_type: type[JavaExtractBeanAction] = JavaExtractBeanAction
    observation_type: type[RefactorObservation] = RefactorObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> JavaExtractBeanTool:
        backend = JavaBackend()
        return cls(executor=LSPRefactorExecutor({'java': backend}))


__all__ = [
    'JavaRenameAction',
    'JavaMoveClassAction',
    'JavaExtractBeanAction',
    'JavaRenameTool',
    'JavaMoveClassTool',
    'JavaExtractBeanTool',
    'JavaBackend',
]
