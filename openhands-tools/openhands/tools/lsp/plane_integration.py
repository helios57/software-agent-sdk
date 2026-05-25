"""Plane project management integration.

Native integration with Plane (open-source Jira/Linear alternative).
Keeps Plane issues in sync with agent work: auto-creates issues for
sub-tasks, transitions state as work progresses, and comments on
important events (PRs opened, tests pass/fail, deployments).

API docs: https://developers.plane.so/api-reference/

Authentication via PLANE_API_KEY env var or passed explicitly.
Base URL via PLANE_BASE_URL (defaults to https://api.plane.so).

Tools:
- plane_create_issue: Create an issue in a Plane project
- plane_update_issue: Update issue state, assignee, priority, description
- plane_comment_issue: Add a comment to an issue
- plane_get_issues: List/filter issues by state, assignee, labels
- plane_create_cycle: Create a sprint/cycle
- plane_sync_task: Sync an OpenHands sub-task to a Plane issue
  (create if not exists, update state, comment with progress)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import requests

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)

PLANE_API_KEY = os.getenv('PLANE_API_KEY', '')
PLANE_BASE_URL = os.getenv('PLANE_BASE_URL', 'https://api.plane.so')
PLANE_WORKSPACE_SLUG = os.getenv('PLANE_WORKSPACE_SLUG', '')


def _plane_headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key or PLANE_API_KEY
    return {
        'x-api-key': key,
        'Content-Type': 'application/json',
    }


def _plane_req(
    method: str, path: str, api_key: str | None = None,
    base_url: str | None = None, **kwargs: Any,
) -> dict[str, Any]:
    """Make an authenticated request to the Plane REST API."""
    url = urljoin(base_url or PLANE_BASE_URL, path)
    resp = requests.request(
        method, url, headers=_plane_headers(api_key),
        timeout=30, **kwargs,
    )
    if not resp.ok:
        logger.error(
            'Plane API %s %s → %d: %s',
            method, path, resp.status_code, resp.text[:500],
        )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ═══════════════════════════════════════════════════════════════════════════════
# Action / Observation schemas
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PlaneCreateIssueAction(Action):
    """Create a new issue in a Plane project."""

    project_id: str
    """UUID of the Plane project."""

    name: str
    """Issue title."""

    description: str = ''
    """Markdown description."""

    priority: str = 'low'
    """'urgent' | 'high' | 'medium' | 'low' | 'none'"""

    state: str = 'backlog'
    """Initial workflow state name or UUID."""

    assignee_ids: list[str] = field(default_factory=list)
    """List of member UUIDs to assign."""

    label_ids: list[str] = field(default_factory=list)
    """List of label UUIDs."""

    cycle_id: str = ''
    """Optional cycle/sprint UUID."""

    start_date: str = ''
    """ISO date string for start date."""

    target_date: str = ''
    """ISO date string for due date."""

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_create_issue'


@dataclass
class PlaneUpdateIssueAction(Action):
    """Update a Plane issue's state or properties."""

    issue_id: str
    """UUID of the issue to update."""

    project_id: str
    """UUID of the parent project."""

    state: str = ''
    """New workflow state name or UUID."""

    name: str = ''
    """New title."""

    description: str = ''
    """New description."""

    priority: str = ''
    """'urgent' | 'high' | 'medium' | 'low' | 'none'"""

    assignee_ids: list[str] = field(default_factory=list)
    cycle_id: str = ''
    target_date: str = ''

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_update_issue'


@dataclass
class PlaneCommentIssueAction(Action):
    """Add a comment to a Plane issue."""

    issue_id: str
    """UUID of the issue."""

    project_id: str
    """UUID of the parent project."""

    body: str
    """Markdown comment body."""

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_comment_issue'


@dataclass
class PlaneGetIssuesAction(Action):
    """List/issues issues in a Plane project."""

    project_id: str
    """UUID of the project."""

    state: str = ''
    """Filter by state name (e.g. 'In Progress', 'Done')."""

    assignee_id: str = ''
    """Filter by assignee UUID."""

    priority: str = ''
    """Filter by priority: 'urgent' | 'high' | 'medium' | 'low'."""

    cycle_id: str = ''
    """Filter by cycle UUID."""

    labels: str = ''
    """Comma-separated label names."""

    search: str = ''
    """Full-text search query."""

    limit: int = 50

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_get_issues'


@dataclass
class PlaneCreateCycleAction(Action):
    """Create a sprint/cycle in a Plane project."""

    project_id: str
    """UUID of the project."""

    name: str
    """Cycle name (e.g. 'Sprint 5')."""

    start_date: str
    """ISO date for start."""

    end_date: str
    """ISO date for end."""

    description: str = ''

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_create_cycle'


@dataclass
class PlaneSyncTaskAction(Action):
    """Sync an OpenHands task to a Plane issue.

    Creates the issue if it doesn't exist (keyed by task_id),
    updates its state to reflect work progress, and optionally
    comments with a status update.
    """

    project_id: str
    """UUID of the Plane project."""

    task_id: str
    """Unique identifier for this task (stable across syncs)."""

    title: str
    """Issue title."""

    state: str = 'backlog'
    """Desired Plane state: 'backlog' | 'unstarted' | 'started'
    | 'in progress' | 'completed' | 'cancelled'."""

    comment: str = ''
    """If non-empty, add this comment to the issue."""

    description: str = ''
    """Issue description (only used on first creation)."""

    priority: str = 'medium'
    assignee_ids: list[str] = field(default_factory=list)
    cycle_id: str = ''

    api_key: str = ''
    base_url: str = ''

    @classmethod
    def name(cls) -> str:
        return 'plane_sync_task'


@dataclass
class PlaneObservation(Observation):
    """Observation from Plane operations."""

    plane_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict[str, Any]) -> PlaneObservation:
        text = '```json\n' + json.dumps(data, indent=2) + '\n```'
        obs = Observation.from_text(text)
        obs.__class__ = cls
        obs.plane_data = data
        obs.is_error = bool(data.get('error'))
        return obs


# ═══════════════════════════════════════════════════════════════════════════════
# Plane utility helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _get_workspace_slug(api_key: str, base_url: str) -> str:
    """Discover workspace slug from the API if not configured."""
    if PLANE_WORKSPACE_SLUG:
        return PLANE_WORKSPACE_SLUG
    try:
        data = _plane_req(
            'GET', '/api/v1/workspaces/', api_key=api_key,
            base_url=base_url,
        )
        return data['results'][0]['slug']  # type: ignore[no-any-return]
    except Exception:
        return ''


def _get_state_id(
    state_name: str, project_id: str, api_key: str, base_url: str,
) -> str | None:
    """Resolve a human-readable state name to a Plane state UUID."""
    if not state_name:
        return None
    try:
        ws = _get_workspace_slug(api_key, base_url)
        data = _plane_req(
            'GET',
            f'/api/v1/workspaces/{ws}/projects/{project_id}/states/',
            api_key=api_key, base_url=base_url,
        )
        for s in data.get('results', []):
            if s.get('name', '').lower() == state_name.lower():
                return s['id']  # type: ignore[no-any-return]
        # Try group match
        for s in data.get('results', []):
            if s.get('group', '').lower() == state_name.lower():
                return s['id']  # type: ignore[no-any-return]
        # Fuzzy: any state containing the name
        for s in data.get('results', []):
            sn = s.get('name', '').lower()
            if state_name.lower() in sn:
                return s['id']  # type: ignore[no-any-return]
        return None
    except Exception as e:
        logger.warning('Could not resolve state %r: %s', state_name, e)
        return None


def _find_issue_by_task_id(
    task_id: str, project_id: str, api_key: str, base_url: str,
) -> dict[str, Any] | None:
    """Find an existing Plane issue keyed by an OpenHands task_id.

    Uses the task_id as a label or searches the description/name.
    """
    try:
        ws = _get_workspace_slug(api_key, base_url)
        # Search by name prefix (task_id embedded)
        data = _plane_req(
            'GET',
            f'/api/v1/workspaces/{ws}/projects/{project_id}/issues/',
            params={'search': task_id, 'limit': '5'},
            api_key=api_key, base_url=base_url,
        )
        results = data.get('results', [])
        # Exact match in name
        for issue in results:
            if task_id in issue.get('name', ''):
                return issue  # type: ignore[no-any-return]
        # Fallback: any result
        if results:
            return results[0]  # type: ignore[no-any-return]
        return None
    except Exception as e:
        logger.warning('Issue lookup for %r failed: %s', task_id, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Create Issue
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneCreateIssueTool(
    ToolDefinition[PlaneCreateIssueAction, PlaneObservation]
):
    """Create a new issue in a Plane project."""

    name: str = 'plane_create_issue'
    description: str = (
        'Create a new issue in a Plane project. Returns the issue ID, '
        'sequence ID (e.g. PROJ-42), and URL. Requires project UUID.'
    )
    action_type: type[PlaneCreateIssueAction] = PlaneCreateIssueAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneCreateIssueTool:
        return cls(executor=PlaneCreateIssueExecutor())


class PlaneCreateIssueExecutor:
    """Create a Plane issue via REST API."""

    def __call__(
        self, action: PlaneCreateIssueAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            state_id = None
            if action.state:
                state_id = _get_state_id(
                    action.state, action.project_id,
                    action.api_key, action.base_url,
                )

            body: dict[str, Any] = {
                'name': action.name,
                'description_html': action.description,
                'priority': action.priority,
            }
            if state_id:
                body['state'] = state_id
            if action.assignee_ids:
                body['assignees_list'] = action.assignee_ids
            if action.label_ids:
                body['labels_list'] = action.label_ids
            if action.cycle_id:
                body['cycle'] = action.cycle_id
            if action.start_date:
                body['start_date'] = action.start_date
            if action.target_date:
                body['target_date'] = action.target_date

            ws = _get_workspace_slug(action.api_key, action.base_url)
            result = _plane_req(
                'POST',
                f'/api/v1/workspaces/{ws}/projects/{action.project_id}/issues/',
                json=body,
                api_key=action.api_key, base_url=action.base_url,
            )

            return PlaneObservation.from_response({
                'action': 'created',
                'issue': {
                    'id': result.get('id'),
                    'sequence_id': result.get('sequence_id'),
                    'name': result.get('name'),
                    'state': result.get('state_detail', {}).get('name', action.state),
                    'url': result.get('url', ''),
                },
            })
        except Exception as e:
            logger.exception('Plane create issue failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Update Issue
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneUpdateIssueTool(
    ToolDefinition[PlaneUpdateIssueAction, PlaneObservation]
):
    """Update a Plane issue's state or properties."""

    name: str = 'plane_update_issue'
    description: str = (
        'Update a Plane issue: change state (backlog → in progress → done), '
        'reassign, change priority, update description. '
        'State names are resolved automatically.'
    )
    action_type: type[PlaneUpdateIssueAction] = PlaneUpdateIssueAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneUpdateIssueTool:
        return cls(executor=PlaneUpdateIssueExecutor())


class PlaneUpdateIssueExecutor:
    """Update a Plane issue via REST API."""

    def __call__(
        self, action: PlaneUpdateIssueAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            body: dict[str, Any] = {}

            if action.state:
                state_id = _get_state_id(
                    action.state, action.project_id,
                    action.api_key, action.base_url,
                )
                if state_id:
                    body['state'] = state_id
            if action.name:
                body['name'] = action.name
            if action.description:
                body['description_html'] = action.description
            if action.priority:
                body['priority'] = action.priority
            if action.assignee_ids:
                body['assignees_list'] = action.assignee_ids
            if action.cycle_id:
                body['cycle'] = action.cycle_id
            if action.target_date:
                body['target_date'] = action.target_date

            if not body:
                return PlaneObservation.from_response({
                    'error': 'No fields to update',
                })

            ws = _get_workspace_slug(action.api_key, action.base_url)
            result = _plane_req(
                'PATCH',
                f'/api/v1/workspaces/{ws}/projects/{action.project_id}'
                f'/issues/{action.issue_id}/',
                json=body,
                api_key=action.api_key, base_url=action.base_url,
            )

            return PlaneObservation.from_response({
                'action': 'updated',
                'issue': {
                    'id': result.get('id'),
                    'sequence_id': result.get('sequence_id'),
                    'state': result.get('state_detail', {}).get(
                        'name', action.state
                    ),
                },
                'changed_fields': list(body.keys()),
            })
        except Exception as e:
            logger.exception('Plane update issue failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Comment Issue
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneCommentIssueTool(
    ToolDefinition[PlaneCommentIssueAction, PlaneObservation]
):
    """Add a comment to a Plane issue."""

    name: str = 'plane_comment_issue'
    description: str = (
        'Add a Markdown comment to a Plane issue. Use this to log '
        'important events: PRs opened, tests passing/failing, '
        'deployments, design decisions.'
    )
    action_type: type[PlaneCommentIssueAction] = PlaneCommentIssueAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneCommentIssueTool:
        return cls(executor=PlaneCommentIssueExecutor())


class PlaneCommentIssueExecutor:
    """Add a comment to a Plane issue via REST API."""

    def __call__(
        self, action: PlaneCommentIssueAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            ws = _get_workspace_slug(action.api_key, action.base_url)
            result = _plane_req(
                'POST',
                f'/api/v1/workspaces/{ws}/projects/{action.project_id}'
                f'/issues/{action.issue_id}/comments/',
                json={'comment_html': action.body},
                api_key=action.api_key, base_url=action.base_url,
            )

            return PlaneObservation.from_response({
                'action': 'commented',
                'comment_id': result.get('id'),
                'issue_id': action.issue_id,
            })
        except Exception as e:
            logger.exception('Plane comment failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Get Issues
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneGetIssuesTool(
    ToolDefinition[PlaneGetIssuesAction, PlaneObservation]
):
    """List/filter issues in a Plane project."""

    name: str = 'plane_get_issues'
    description: str = (
        'List issues in a Plane project with optional filters by state, '
        'assignee, priority, cycle, labels, and full-text search. '
        'Returns issue IDs, sequence IDs, titles, states, and URLs.'
    )
    action_type: type[PlaneGetIssuesAction] = PlaneGetIssuesAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneGetIssuesTool:
        return cls(executor=PlaneGetIssuesExecutor())


class PlaneGetIssuesExecutor:
    """Fetch and filter Plane issues."""

    def __call__(
        self, action: PlaneGetIssuesAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            ws = _get_workspace_slug(action.api_key, action.base_url)
            params: dict[str, Any] = {'limit': str(action.limit)}

            if action.state:
                state_id = _get_state_id(
                    action.state, action.project_id,
                    action.api_key, action.base_url,
                )
                if state_id:
                    params['state'] = state_id
            if action.assignee_id:
                params['assignees'] = action.assignee_id
            if action.priority:
                params['priority'] = action.priority
            if action.cycle_id:
                params['cycle'] = action.cycle_id
            if action.labels:
                params['labels'] = action.labels
            if action.search:
                params['search'] = action.search

            data = _plane_req(
                'GET',
                f'/api/v1/workspaces/{ws}/projects/{action.project_id}/issues/',
                params=params,
                api_key=action.api_key, base_url=action.base_url,
            )

            issues = []
            for issue in data.get('results', []):
                issues.append({
                    'id': issue.get('id'),
                    'sequence_id': issue.get('sequence_id'),
                    'name': issue.get('name'),
                    'state': issue.get('state_detail', {}).get(
                        'name', 'unknown'
                    ),
                    'priority': issue.get('priority', 'none'),
                    'url': issue.get('url', ''),
                })

            return PlaneObservation.from_response({
                'project_id': action.project_id,
                'total': data.get('total_count', len(issues)),
                'issues': issues,
            })
        except Exception as e:
            logger.exception('Plane get issues failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Create Cycle
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneCreateCycleTool(
    ToolDefinition[PlaneCreateCycleAction, PlaneObservation]
):
    """Create a sprint/cycle in a Plane project."""

    name: str = 'plane_create_cycle'
    description: str = (
        'Create a sprint/cycle in a Plane project. Returns the cycle ID '
        'which can be used when creating or updating issues.'
    )
    action_type: type[PlaneCreateCycleAction] = PlaneCreateCycleAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneCreateCycleTool:
        return cls(executor=PlaneCreateCycleExecutor())


class PlaneCreateCycleExecutor:
    """Create a Plane cycle via REST API."""

    def __call__(
        self, action: PlaneCreateCycleAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            ws = _get_workspace_slug(action.api_key, action.base_url)
            result = _plane_req(
                'POST',
                f'/api/v1/workspaces/{ws}/projects/{action.project_id}/cycles/',
                json={
                    'name': action.name,
                    'start_date': action.start_date,
                    'end_date': action.end_date,
                    'description': action.description,
                },
                api_key=action.api_key, base_url=action.base_url,
            )

            return PlaneObservation.from_response({
                'action': 'created',
                'cycle': {
                    'id': result.get('id'),
                    'name': result.get('name'),
                    'start_date': result.get('start_date'),
                    'end_date': result.get('end_date'),
                },
            })
        except Exception as e:
            logger.exception('Plane create cycle failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Sync Task  (the main integration workhorse)
# ═══════════════════════════════════════════════════════════════════════════════


class PlaneSyncTaskTool(
    ToolDefinition[PlaneSyncTaskAction, PlaneObservation]
):
    """Sync an OpenHands task to a Plane issue.

    This is the primary integration tool: call it whenever work on a
    task changes state. It creates the issue if it doesn't exist yet
    (keyed by task_id in description), updates the state to reflect
    progress, and optionally adds a comment.
    """

    name: str = 'plane_sync_task'
    description: str = (
        'Sync an OpenHands task to a Plane issue. Creates the issue if '
        'it does not exist, updates its workflow state, and optionally '
        'adds a progress comment. Call this whenever work on a task '
        'changes: started, PR opened, tests passing, deployed, done. '
        'States: backlog, unstarted, started, in progress, completed, cancelled.'
    )
    action_type: type[PlaneSyncTaskAction] = PlaneSyncTaskAction
    observation_type: type[PlaneObservation] = PlaneObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=True, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> PlaneSyncTaskTool:
        return cls(executor=PlaneSyncTaskExecutor())


class PlaneSyncTaskExecutor:
    """Create-or-update Plane issue for a task, auto-commenting on state changes."""

    def __call__(
        self, action: PlaneSyncTaskAction, conversation: Any,
    ) -> PlaneObservation:
        try:
            existing = _find_issue_by_task_id(
                action.task_id, action.project_id,
                action.api_key, action.base_url,
            )

            if existing is None:
                # ── CREATE ──
                ws = _get_workspace_slug(action.api_key, action.base_url)
                state_id = _get_state_id(
                    action.state, action.project_id,
                    action.api_key, action.base_url,
                )

                body: dict[str, Any] = {
                    'name': f'{action.title}',
                    'description_html': (
                        f'{action.description}\n\n'
                        f'<!-- openhands_task_id: {action.task_id} -->'
                    ),
                    'priority': action.priority,
                }
                if state_id:
                    body['state'] = state_id
                if action.assignee_ids:
                    body['assignees_list'] = action.assignee_ids
                if action.cycle_id:
                    body['cycle'] = action.cycle_id

                result = _plane_req(
                    'POST',
                    f'/api/v1/workspaces/{ws}/projects/{action.project_id}/issues/',
                    json=body,
                    api_key=action.api_key, base_url=action.base_url,
                )
                issue_id = result['id']
                sequence_id = result.get('sequence_id', '?')

                # Auto-comment on creation
                if action.comment:
                    _plane_req(
                        'POST',
                        f'/api/v1/workspaces/{ws}/projects/{action.project_id}'
                        f'/issues/{issue_id}/comments/',
                        json={'comment_html': action.comment},
                        api_key=action.api_key, base_url=action.base_url,
                    )

                return PlaneObservation.from_response({
                    'action': 'created',
                    'issue_id': issue_id,
                    'sequence_id': sequence_id,
                    'state': action.state,
                    'commented': bool(action.comment),
                })

            # ── UPDATE ──
            ws = _get_workspace_slug(action.api_key, action.base_url)
            issue_id = existing['id']
            old_state = existing.get('state_detail', {}).get('name', '')

            body: dict[str, Any] = {}
            state_changed = False

            if action.state and action.state.lower() != old_state.lower():
                state_id = _get_state_id(
                    action.state, action.project_id,
                    action.api_key, action.base_url,
                )
                if state_id:
                    body['state'] = state_id
                    state_changed = True

            if action.priority:
                body['priority'] = action.priority

            if body:
                _plane_req(
                    'PATCH',
                    f'/api/v1/workspaces/{ws}/projects/{action.project_id}'
                    f'/issues/{issue_id}/',
                    json=body,
                    api_key=action.api_key, base_url=action.base_url,
                )

            # Auto-comment when state changes
            if state_changed and action.comment:
                _plane_req(
                    'POST',
                    f'/api/v1/workspaces/{ws}/projects/{action.project_id}'
                    f'/issues/{issue_id}/comments/',
                    json={'comment_html': action.comment},
                    api_key=action.api_key, base_url=action.base_url,
                )

            return PlaneObservation.from_response({
                'action': 'synced',
                'issue_id': issue_id,
                'sequence_id': existing.get('sequence_id', '?'),
                'old_state': old_state,
                'new_state': action.state if state_changed else old_state,
                'state_changed': state_changed,
                'commented': bool(state_changed and action.comment),
            })

        except Exception as e:
            logger.exception('Plane sync task failed')
            return PlaneObservation.from_response({'error': str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Registration & Exports
# ═══════════════════════════════════════════════════════════════════════════════


def register_plane_tools() -> list[type[ToolDefinition]]:
    """Return all Plane integration tool definitions."""
    return [
        PlaneCreateIssueTool,
        PlaneUpdateIssueTool,
        PlaneCommentIssueTool,
        PlaneGetIssuesTool,
        PlaneCreateCycleTool,
        PlaneSyncTaskTool,
    ]


__all__ = [
    # Actions
    'PlaneCreateIssueAction',
    'PlaneUpdateIssueAction',
    'PlaneCommentIssueAction',
    'PlaneGetIssuesAction',
    'PlaneCreateCycleAction',
    'PlaneSyncTaskAction',
    # Tools
    'PlaneCreateIssueTool',
    'PlaneUpdateIssueTool',
    'PlaneCommentIssueTool',
    'PlaneGetIssuesTool',
    'PlaneCreateCycleTool',
    'PlaneSyncTaskTool',
    # Registry
    'register_plane_tools',
]