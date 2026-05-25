"""LSP/AST refactoring tools for OpenHands agent.

Provides language-aware refactoring tools that use native LSP/AST backends
instead of regex/string replacement. The LLM decides what to refactor;
the native tooling executes precise AST mutations.

Tools by priority:
  P1 — Python: py_rename, py_extract_method, py_inline_variable
  P2 — Go:     go_rename, go_extract_interface, go_extract_function
  P3 — TS:     ts_rename_symbol, ts_extract_component, ts_move_file
  P4 — Java:   java_safe_rename, java_move_class, java_extract_bean
  P5 — Cross:  cross_boundary_refactor
  P6 — Effic:  get_call_graph, find_implementations, explain_dead_code

Force Multipliers:
  FM1 — Memory:    semantic_checkpoint, semantic_recall
  FM2 — Roles:     decompose_task, validate_contract,
                   review_specialist, test_specialist, e2e_user_test
  FM3 — ArgoCD:    argocd_health_check
  FM4 — Tests:     test_gen, test_refine

Integrations:
  Plane — plane_create_issue, plane_update_issue, plane_comment_issue,
          plane_get_issues, plane_create_cycle, plane_sync_task

Activation:
    from openhands.tools.lsp import register_lsp_tools
    register_lsp_tools()
"""

from __future__ import annotations

import logging

from openhands.sdk.tool.registry import register_tool

logger = logging.getLogger(__name__)


def register_lsp_tools() -> None:
    """Register all LSP/AST refactoring tools with the global tool registry.

    Call this early in agent initialization (e.g., from
    register_default_tools()) to make the tools available to all agents.
    """
    # ── P1: Python tools ─────────────────────────────────────────────────
    from openhands.tools.lsp.python import (
        PyRenameTool,
        PyExtractMethodTool,
        PyInlineVariableTool,
    )

    register_tool(PyRenameTool.name, PyRenameTool)
    register_tool(PyExtractMethodTool.name, PyExtractMethodTool)
    register_tool(PyInlineVariableTool.name, PyInlineVariableTool)

    # ── P2: Go tools ─────────────────────────────────────────────────────
    from openhands.tools.lsp.go import (
        GoRenameTool,
        GoExtractInterfaceTool,
        GoExtractFunctionTool,
    )

    register_tool(GoRenameTool.name, GoRenameTool)
    register_tool(GoExtractInterfaceTool.name, GoExtractInterfaceTool)
    register_tool(GoExtractFunctionTool.name, GoExtractFunctionTool)

    # ── P3: TypeScript tools ─────────────────────────────────────────────
    from openhands.tools.lsp.typescript import (
        TSRenameSymbolTool,
        TSExtractComponentTool,
        TSMoveFileTool,
    )

    register_tool(TSRenameSymbolTool.name, TSRenameSymbolTool)
    register_tool(TSExtractComponentTool.name, TSExtractComponentTool)
    register_tool(TSMoveFileTool.name, TSMoveFileTool)

    # ── P4: Java tools ───────────────────────────────────────────────────
    from openhands.tools.lsp.java import (
        JavaRenameTool,
        JavaMoveClassTool,
        JavaExtractBeanTool,
    )

    register_tool(JavaRenameTool.name, JavaRenameTool)
    register_tool(JavaMoveClassTool.name, JavaMoveClassTool)
    register_tool(JavaExtractBeanTool.name, JavaExtractBeanTool)

    # ── P5: Cross-boundary refactor ──────────────────────────────────────
    from openhands.tools.lsp.cross_boundary import CrossBoundaryRefactorTool

    register_tool(CrossBoundaryRefactorTool.name, CrossBoundaryRefactorTool)

    # ── P6: Efficiency tools ─────────────────────────────────────────────
    from openhands.tools.lsp.efficiency import (
        GetCallGraphTool,
        FindImplementationsTool,
        ExplainDeadCodeTool,
    )

    register_tool(GetCallGraphTool.name, GetCallGraphTool)
    register_tool(FindImplementationsTool.name, FindImplementationsTool)
    register_tool(ExplainDeadCodeTool.name, ExplainDeadCodeTool)

    # ── FM1: Memory layer ────────────────────────────────────────────────
    from openhands.tools.lsp.memory_layer import (
        SemanticCheckpointTool,
        SemanticRecallTool,
    )

    register_tool(SemanticCheckpointTool.name, SemanticCheckpointTool)
    register_tool(SemanticRecallTool.name, SemanticRecallTool)

    # ── FM2: Role-based agents ───────────────────────────────────────────
    from openhands.tools.lsp.role_agents import (
        DecomposeTaskTool,
        ValidateContractTool,
        ReviewSpecialistTool,
        TestSpecialistTool,
        E2EUserTestTool,
    )

    register_tool(DecomposeTaskTool.name, DecomposeTaskTool)
    register_tool(ValidateContractTool.name, ValidateContractTool)
    register_tool(ReviewSpecialistTool.name, ReviewSpecialistTool)
    register_tool(TestSpecialistTool.name, TestSpecialistTool)
    register_tool(E2EUserTestTool.name, E2EUserTestTool)

    # ── FM3: ArgoCD health ───────────────────────────────────────────────
    from openhands.tools.lsp.argo_health import ArgoCDHealthCheckTool

    register_tool(ArgoCDHealthCheckTool.name, ArgoCDHealthCheckTool)

    # ── FM4: Test generation/refinement ──────────────────────────────────
    from openhands.tools.lsp.test_autogen import (
        TestGenTool,
        TestRefineTool,
    )

    register_tool(TestGenTool.name, TestGenTool)
    register_tool(TestRefineTool.name, TestRefineTool)

    # ── Plane integration ────────────────────────────────────────────────
    from openhands.tools.lsp.plane_integration import (
        PlaneCreateIssueTool,
        PlaneUpdateIssueTool,
        PlaneCommentIssueTool,
        PlaneGetIssuesTool,
        PlaneCreateCycleTool,
        PlaneSyncTaskTool,
    )

    register_tool(PlaneCreateIssueTool.name, PlaneCreateIssueTool)
    register_tool(PlaneUpdateIssueTool.name, PlaneUpdateIssueTool)
    register_tool(PlaneCommentIssueTool.name, PlaneCommentIssueTool)
    register_tool(PlaneGetIssuesTool.name, PlaneGetIssuesTool)
    register_tool(PlaneCreateCycleTool.name, PlaneCreateCycleTool)
    register_tool(PlaneSyncTaskTool.name, PlaneSyncTaskTool)

    logger.info('Registered %d LSP/AST refactoring tools', 31)


# List all tool names for discovery
ALL_LSP_TOOLS = [
    # Python
    'py_rename', 'py_extract_method', 'py_inline_variable',
    # Go
    'go_rename', 'go_extract_interface', 'go_extract_function',
    # TypeScript
    'ts_rename_symbol', 'ts_extract_component', 'ts_move_file',
    # Java
    'java_safe_rename', 'java_move_class', 'java_extract_bean',
    # Cross-boundary
    'cross_boundary_refactor',
    # Efficiency
    'get_call_graph', 'find_implementations', 'explain_dead_code',
    # Force multipliers
    'semantic_checkpoint', 'semantic_recall',
    'decompose_task', 'validate_contract',
    'review_specialist', 'test_specialist', 'e2e_user_test',
    'argocd_health_check',
    'test_gen', 'test_refine',
    # Plane integration
    'plane_create_issue', 'plane_update_issue', 'plane_comment_issue',
    'plane_get_issues', 'plane_create_cycle', 'plane_sync_task',
]

__all__ = [
    'register_lsp_tools',
    'ALL_LSP_TOOLS',
]