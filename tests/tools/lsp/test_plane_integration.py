"""Tests for Plane integration — mocked HTTP calls."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openhands.tools.lsp.plane_integration import (
    PlaneCreateIssueAction,
    PlaneCreateIssueExecutor,
    PlaneUpdateIssueAction,
    PlaneUpdateIssueExecutor,
    PlaneCommentIssueAction,
    PlaneCommentIssueExecutor,
    PlaneGetIssuesAction,
    PlaneGetIssuesExecutor,
    PlaneCreateCycleAction,
    PlaneCreateCycleExecutor,
    PlaneSyncTaskAction,
    PlaneSyncTaskExecutor,
    _get_state_id,
    _find_issue_by_task_id,
    _get_workspace_slug,
)


def _empty_list() -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# State resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateIdResolution:
    """Test state name → UUID resolution."""

    def test_resolve_by_name(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'results': [
                    {'id': 'uuid-1', 'name': 'Backlog', 'group': 'backlog'},
                    {'id': 'uuid-2', 'name': 'In Progress', 'group': 'started'},
                    {'id': 'uuid-3', 'name': 'Done', 'group': 'completed'},
                ],
            },
        ):
            sid = _get_state_id('In Progress', 'proj-1', '', '')
            assert sid == 'uuid-2'

    def test_resolve_case_insensitive(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'results': [
                    {'id': 'uuid-1', 'name': 'Backlog'},
                ],
            },
        ):
            sid = _get_state_id('backlog', 'proj-1', '', '')
            assert sid == 'uuid-1'

    def test_resolve_not_found(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={'results': []},
        ):
            sid = _get_state_id('NonExistent', 'proj-1', '', '')
            assert sid is None

    def test_resolve_empty_string(self):
        sid = _get_state_id('', 'proj-1', '', '')
        assert sid is None


# ═══════════════════════════════════════════════════════════════════════════════
# Find issue by task_id
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindIssueByTaskId:
    """Test issue lookup by task_id."""

    def test_finds_exact_match(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'results': [
                    {'id': 'issue-1', 'name': '[task-abc] Fix login',
                     'sequence_id': 'PROJ-42'},
                ],
            },
        ):
            issue = _find_issue_by_task_id('task-abc', 'proj-1', '', '')
            assert issue is not None
            assert issue['id'] == 'issue-1'

    def test_no_match_returns_none(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={'results': []},
        ):
            issue = _find_issue_by_task_id('task-xyz', 'proj-1', '', '')
            assert issue is None

    def test_exception_returns_none(self):
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            side_effect=Exception('network error'),
        ):
            issue = _find_issue_by_task_id('task-abc', 'proj-1', '', '')
            assert issue is None


# ═══════════════════════════════════════════════════════════════════════════════
# Workspace slug resolution
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceSlug:
    """Test workspace slug discovery."""

    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setattr(
            'openhands.tools.lsp.plane_integration.PLANE_WORKSPACE_SLUG',
            'my-workspace',
        )
        slug = _get_workspace_slug('', '')
        assert slug == 'my-workspace'

    def test_falls_back_to_api(self, monkeypatch):
        monkeypatch.setattr(
            'openhands.tools.lsp.plane_integration.PLANE_WORKSPACE_SLUG', ''
        )
        with patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={'results': [{'slug': 'api-workspace'}]},
        ):
            slug = _get_workspace_slug('key', 'https://plane.example.com')
            assert slug == 'api-workspace'


# ═══════════════════════════════════════════════════════════════════════════════
# Sync task executor
# ═══════════════════════════════════════════════════════════════════════════════


class TestSyncTaskExecutor:
    """Test the plane_sync_task create-or-update logic."""

    def _make_action(self, **overrides):
        defaults = {
            'project_id': 'proj-1', 'task_id': 'task-001',
            'title': 'Test issue', 'state': 'backlog',
            'comment': '', 'description': '',
            'priority': 'medium', 'assignee_ids': [],
            'cycle_id': '', 'api_key': '', 'base_url': '',
        }
        return PlaneSyncTaskAction.model_construct(**{**defaults, **overrides})

    def test_creates_new_issue_when_not_found(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._find_issue_by_task_id',
            return_value=None,
        ), patch(
            'openhands.tools.lsp.plane_integration._get_state_id',
            return_value='state-uuid',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
        ) as mock_req:
            mock_req.side_effect = [
                {'id': 'new-issue', 'sequence_id': 'PROJ-99',
                 'state_detail': {'name': 'Backlog'}},
            ]

            executor = PlaneSyncTaskExecutor()
            action = self._make_action(description='Test description')
            result = executor(action, conversation=None)

            assert result.plane_data['action'] == 'created'
            assert result.plane_data['issue_id'] == 'new-issue'
            assert result.plane_data['sequence_id'] == 'PROJ-99'

    def test_updates_existing_issue(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._find_issue_by_task_id',
            return_value={
                'id': 'existing-issue',
                'sequence_id': 'PROJ-42',
                'state_detail': {'name': 'Backlog'},
            },
        ), patch(
            'openhands.tools.lsp.plane_integration._get_state_id',
            return_value='in-progress-uuid',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
        ) as mock_req:
            mock_req.return_value = {
                'id': 'existing-issue', 'sequence_id': 'PROJ-42',
                'state_detail': {'name': 'In Progress'},
            }

            executor = PlaneSyncTaskExecutor()
            action = self._make_action(
                state='in progress',
                comment='Moving to in progress',
            )
            result = executor(action, conversation=None)

            assert result.plane_data['action'] == 'synced'
            assert result.plane_data['state_changed'] is True
            assert result.plane_data['old_state'] == 'Backlog'
            assert result.plane_data['new_state'] == 'in progress'

    def test_no_state_change_when_same_state(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._find_issue_by_task_id',
            return_value={
                'id': 'issue-1',
                'sequence_id': 'PROJ-1',
                'state_detail': {'name': 'In Progress'},
            },
        ), patch(
            'openhands.tools.lsp.plane_integration._get_state_id',
            return_value='in-progress-uuid',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
        ):
            executor = PlaneSyncTaskExecutor()
            action = self._make_action(
                state='in progress',  # same as current
            )
            result = executor(action, conversation=None)

            assert result.plane_data['state_changed'] is False

    def test_handles_api_error_gracefully(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._find_issue_by_task_id',
            side_effect=Exception('Network down'),
        ):
            executor = PlaneSyncTaskExecutor()
            action = self._make_action()
            result = executor(action, conversation=None)
            assert result.is_error is True
            assert 'Network down' in result.plane_data.get('error', '')


# ═══════════════════════════════════════════════════════════════════════════════
# Create / Update / Comment / Get / Cycle executors
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlaneCrudExecutors:
    """Test create, update, comment, get, and cycle executors."""

    def test_create_issue(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._get_state_id',
            return_value=None,
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'id': 'issue-1', 'sequence_id': 'PROJ-10',
                'name': 'My Issue',
                'state_detail': {'name': 'Backlog'},
                'url': 'https://plane.example.com/ws/proj/issues/PROJ-10',
            },
        ):
            executor = PlaneCreateIssueExecutor()
            action = PlaneCreateIssueAction.model_construct(
                project_id='proj-1', name='My Issue',
                description='Test', priority='medium',
                state='backlog', assignee_ids=[], label_ids=[],
                cycle_id='', start_date='', target_date='',
                api_key='', base_url='',
            )
            result = executor(action, conversation=None)
            assert result.plane_data['action'] == 'created'
            assert result.plane_data['issue']['id'] == 'issue-1'

    def test_update_issue(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._get_state_id',
            return_value='done-uuid',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'id': 'issue-1', 'sequence_id': 'PROJ-10',
                'state_detail': {'name': 'Done'},
            },
        ):
            executor = PlaneUpdateIssueExecutor()
            action = PlaneUpdateIssueAction.model_construct(
                issue_id='issue-1', project_id='proj-1',
                state='done', name='', description='', priority='',
                assignee_ids=[], cycle_id='', target_date='',
                api_key='', base_url='',
            )
            result = executor(action, conversation=None)
            assert result.plane_data['action'] == 'updated'

    def test_update_empty_returns_error(self):
        executor = PlaneUpdateIssueExecutor()
        action = PlaneUpdateIssueAction.model_construct(
            issue_id='x', project_id='proj-1',
            state='', name='', description='', priority='',
            assignee_ids=[], cycle_id='', target_date='',
            api_key='', base_url='',
        )
        result = executor(action, conversation=None)
        assert result.is_error is True
        assert 'No fields' in result.plane_data.get('error', '')

    def test_comment_issue(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={'id': 'comment-1'},
        ):
            executor = PlaneCommentIssueExecutor()
            action = PlaneCommentIssueAction.model_construct(
                issue_id='issue-1', project_id='proj-1',
                body='PR opened: #42', api_key='', base_url='',
            )
            result = executor(action, conversation=None)
            assert result.plane_data['action'] == 'commented'

    def test_get_issues(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'total_count': 2,
                'results': [
                    {'id': 'i1', 'sequence_id': 'PROJ-1',
                     'name': 'Issue 1',
                     'state_detail': {'name': 'Backlog'},
                     'priority': 'high', 'url': ''},
                    {'id': 'i2', 'sequence_id': 'PROJ-2',
                     'name': 'Issue 2',
                     'state_detail': {'name': 'In Progress'},
                     'priority': 'low', 'url': ''},
                ],
            },
        ):
            executor = PlaneGetIssuesExecutor()
            action = PlaneGetIssuesAction.model_construct(
                project_id='proj-1', state='', assignee_id='',
                priority='', cycle_id='', labels='', search='',
                limit=10, api_key='', base_url='',
            )
            result = executor(action, conversation=None)
            assert result.plane_data['total'] == 2
            assert len(result.plane_data['issues']) == 2

    def test_create_cycle(self):
        with patch(
            'openhands.tools.lsp.plane_integration._get_workspace_slug',
            return_value='ws',
        ), patch(
            'openhands.tools.lsp.plane_integration._plane_req',
            return_value={
                'id': 'cycle-1', 'name': 'Sprint 1',
                'start_date': '2026-01-01', 'end_date': '2026-01-14',
            },
        ):
            executor = PlaneCreateCycleExecutor()
            action = PlaneCreateCycleAction.model_construct(
                project_id='proj-1', name='Sprint 1',
                start_date='2026-01-01', end_date='2026-01-14',
                description='', api_key='', base_url='',
            )
            result = executor(action, conversation=None)
            assert result.plane_data['action'] == 'created'
            assert result.plane_data['cycle']['id'] == 'cycle-1'
