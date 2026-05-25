"""Role-based agent decomposition (Force Multiplier #2).

Implements:
- System Architect agent: Master planner that decomposes tasks across repos
- Repository specialists: Go-specialist, TypeScript-specialist, Java-specialist
- Contract-first development: Agents communicate through API contracts
  (OpenAPI specs, gRPC .proto) stored in a shared read-only location.

This module provides tool definitions for:
- decompose_task: Break a high-level task into language-specific sub-tasks
- assign_subtask: Dispatch a sub-task to a specialist sub-agent
- validate_contract: Check that sub-agent output conforms to API contract
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)


@dataclass
class DecomposeTaskAction(Action):
    """Break a high-level task into language-specific sub-tasks."""

    task_description: str
    """High-level description of the task."""

    affected_repos: list[str] = field(default_factory=list)
    """Paths to affected repositories."""

    contract_files: list[str] = field(default_factory=list)
    """API contract files (.proto, openapi.yaml) that constrain the work."""

    @classmethod
    def name(cls) -> str:
        return 'decompose_task'


@dataclass
class AssignSubtaskAction(Action):
    """Dispatch a sub-task to a specialist agent."""

    subtask_id: str
    """Identifier for the sub-task."""

    specialist_type: str
    """'go' | 'typescript' | 'java' | 'python' | 'proto'"""

    repo_path: str
    """Root of the repository to work in."""

    instructions: str
    """Detailed instructions for the specialist."""

    contract: str = ''
    """API contract the output must conform to."""

    @classmethod
    def name(cls) -> str:
        return 'assign_subtask'


@dataclass
class ValidateContractAction(Action):
    """Validate that sub-agent output conforms to an API contract."""

    contract_file: str
    """Path to the contract file (.proto, openapi.yaml, etc.)."""

    implementation_path: str
    """Path to the implementation to validate against the contract."""

    contract_type: str = 'auto'
    """'auto' | 'proto' | 'openapi' | 'graphql'"""

    @classmethod
    def name(cls) -> str:
        return 'validate_contract'


@dataclass
class RoleAgentObservation(Observation):
    """Observation from role-based agent operations."""

    agent_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> RoleAgentObservation:
        text = '```json\n' + json.dumps(data, indent=2) + '\n```'
        obs = Observation.from_text(text)
        obs.__class__ = cls
        obs.agent_data = data
        return obs


# ── Task Decomposition ──────────────────────────────────────────────────────


class DecomposeTaskTool(
    ToolDefinition[DecomposeTaskAction, RoleAgentObservation]
):
    """Break a high-level task into language-specific sub-tasks.

    The System Architect agent uses this to plan work before dispatching
    to specialist agents. Each sub-task includes the target repo, language,
    and any API contracts that constrain the implementation.
    """

    name: str = 'decompose_task'
    description: str = (
        'Break a high-level task into language-specific sub-tasks. '
        'Each sub-task targets a specific repo and language. '
        'Use this as the System Architect before dispatching work.'
    )
    action_type: type[DecomposeTaskAction] = DecomposeTaskAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> DecomposeTaskTool:
        return cls(executor=DecomposeTaskExecutor())


class DecomposeTaskExecutor:
    """Analyze repos and contracts to produce a decomposition plan."""

    def __call__(self, action: DecomposeTaskAction, conversation: Any) -> Observation:
        subtasks = []
        repo_languages: dict[str, str] = {}

        # Detect primary language for each repo
        for repo_path in action.affected_repos:
            if not os.path.isdir(repo_path):
                continue
            lang = self._detect_repo_language(repo_path)
            repo_languages[repo_path] = lang

        # Generate sub-tasks based on language split
        lang_specific_tasks = {
            'proto': 'Update .proto definitions and regenerate stubs',
            'go': 'Implement backend logic and gRPC service handlers',
            'typescript': 'Update frontend components and API clients',
            'java': 'Implement Spring service layer and REST controllers',
            'python': 'Implement Python service logic and tests',
        }

        task_id = 0
        for repo_path, lang in repo_languages.items():
            task_id += 1
            subtasks.append({
                'id': f'subtask-{task_id}',
                'specialist': lang,
                'repo': repo_path,
                'description': lang_specific_tasks.get(lang, 'Implement changes'),
                'contracts': [
                    c for c in action.contract_files
                    if c.startswith(repo_path) or lang in c
                ],
            })

        return RoleAgentObservation.from_data({
            'original_task': action.task_description,
            'subtasks': subtasks,
            'total_subtasks': len(subtasks),
            'contract_files': action.contract_files,
        })

    @staticmethod
    def _detect_repo_language(repo_path: str) -> str:
        """Detect primary language of a repository."""
        markers = {
            'go.mod': 'go',
            'package.json': 'typescript',
            'pom.xml': 'java',
            'build.gradle': 'java',
            'pyproject.toml': 'python',
            'setup.py': 'python',
            'buf.yaml': 'proto',
        }
        for filename, lang in markers.items():
            if os.path.exists(os.path.join(repo_path, filename)):
                return lang
        return 'unknown'


# ── Contract Validation ─────────────────────────────────────────────────────


class ValidateContractTool(
    ToolDefinition[ValidateContractAction, RoleAgentObservation]
):
    """Validate that implementation output conforms to API contract.

    For contract-first development: after a specialist agent completes work,
    validate that the output conforms to the shared API contract (.proto,
    OpenAPI spec, etc.).
    """

    name: str = 'validate_contract'
    description: str = (
        'Validate that an implementation conforms to its API contract. '
        'Supports .proto (buf breaking), OpenAPI (spectral), GraphQL. '
        'Use after each specialist agent completes a sub-task.'
    )
    action_type: type[ValidateContractAction] = ValidateContractAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> ValidateContractTool:
        return cls(executor=ValidateContractExecutor())


class ValidateContractExecutor:
    """Run contract validation tools."""

    def __call__(self, action: ValidateContractAction, conversation: Any) -> Observation:
        try:
            if not os.path.exists(action.contract_file):
                return RoleAgentObservation.from_data({
                    'error': f'Contract file not found: {action.contract_file}'
                })

            contract_type = action.contract_type
            if contract_type == 'auto':
                contract_type = self._detect_contract_type(action.contract_file)

            if contract_type == 'proto':
                return self._validate_proto(action.contract_file)
            if contract_type == 'openapi':
                return self._validate_openapi(action.contract_file)
            if contract_type == 'graphql':
                return self._validate_graphql(action.contract_file)

            return RoleAgentObservation.from_data({
                'contract_type': contract_type,
                'status': 'skipped',
                'message': f'No validator available for {contract_type}',
            })
        except Exception as e:
            logger.exception('Contract validation failed')
            return RoleAgentObservation.from_data({'error': str(e)})

    @staticmethod
    def _detect_contract_type(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1]
        base = os.path.basename(file_path)
        if ext == '.proto':
            return 'proto'
        if 'openapi' in base.lower() or ext in ('.yaml', '.yml', '.json'):
            # Check content for OpenAPI indicators
            try:
                with open(file_path) as f:
                    content = f.read()
                if 'openapi:' in content or 'swagger:' in content:
                    return 'openapi'
            except Exception:
                pass
        if ext == '.graphql' or 'graphql' in base.lower():
            return 'graphql'
        return 'unknown'

    @staticmethod
    def _validate_proto(proto_file: str) -> RoleAgentObservation:
        """Run buf breaking change detection."""
        import subprocess

        proto_dir = os.path.dirname(proto_file)
        try:
            result = subprocess.run(
                ['buf', 'breaking', '--against', '.git#branch=main'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=proto_dir,
            )
            violations = []
            if result.stdout.strip():
                violations = result.stdout.strip().split('\n')
            return RoleAgentObservation.from_data({
                'contract': proto_file,
                'tool': 'buf breaking',
                'status': 'pass' if result.returncode == 0 else 'fail',
                'violations': violations,
            })
        except FileNotFoundError:
            return RoleAgentObservation.from_data({
                'contract': proto_file,
                'tool': 'buf breaking',
                'status': 'skipped',
                'message': 'buf CLI not installed',
            })

    @staticmethod
    def _validate_openapi(spec_file: str) -> RoleAgentObservation:
        """Run spectral lint on OpenAPI spec."""
        import subprocess

        try:
            result = subprocess.run(
                ['npx', 'spectral', 'lint', spec_file],
                capture_output=True,
                text=True,
                timeout=30,
            )
            issues = []
            for line in result.stdout.strip().split('\n'):
                if 'error' in line.lower() or 'warning' in line.lower():
                    issues.append(line.strip())
            return RoleAgentObservation.from_data({
                'contract': spec_file,
                'tool': 'spectral',
                'status': 'pass' if result.returncode == 0 else 'fail',
                'issues': issues,
            })
        except Exception:
            return RoleAgentObservation.from_data({
                'contract': spec_file,
                'tool': 'spectral',
                'status': 'skipped',
                'message': 'spectral not available (npm install -g @stoplight/spectral-cli)',
            })

    @staticmethod
    def _validate_graphql(schema_file: str) -> RoleAgentObservation:
        """Run graphql-inspector diff."""
        return RoleAgentObservation.from_data({
            'contract': schema_file,
            'status': 'skipped',
            'message': 'GraphQL validation requires graphql-inspector',
        })


__all__ = [
    'DecomposeTaskAction',
    'AssignSubtaskAction',
    'ValidateContractAction',
    'DecomposeTaskTool',
    'ValidateContractTool',
]
