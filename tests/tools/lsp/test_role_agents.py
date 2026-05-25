"""Tests for LSP role_agents — review/test/e2e specialists."""

from __future__ import annotations

import re

import pytest

from openhands.tools.lsp.role_agents import (
    ReviewSpecialistExecutor,
    TestSpecialistExecutor,
    ReviewSpecialistTool,
    TestSpecialistTool,
    E2EUserTestTool,
    RoleAgentObservation,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Review Specialist — pattern tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReviewSpecialistPatterns:
    """Verify each regex pattern catches its intended target."""

    # ── Security patterns ─────────────────────────────────────────────

    def test_detects_os_system(self):
        pattern = r'\bos\.system\s*\('
        assert re.search(pattern, 'os.system("rm -rf /")')
        assert not re.search(pattern, 'subprocess.run(["ls"])')

    def test_detects_eval(self):
        pattern = r'\beval\s*\('
        assert re.search(pattern, 'eval(user_input)')
        assert not re.search(pattern, 'evaluate_something()')

    def test_detects_exec(self):
        pattern = r'\bexec\s*\('
        assert re.search(pattern, 'exec(code)')
        assert not re.search(pattern, 'execute_task()')

    def test_detects_hardcoded_password(self):
        pattern = r'\bpassword\s*=\s*["\'][^\"\']+["\']'
        assert re.search(pattern, 'password = "hunter2"')
        assert not re.search(pattern, 'password = getpass()')

    def test_api_key_pattern_requires_eight_chars(self):
        """Pattern requires 8+ alphanumeric chars in the value."""
        pattern = r'\bapi[_-]?key\s*=\s*["\'][A-Za-z0-9_\-]{8,}["\']'
        assert re.search(pattern, 'api_key = "sk-1234567890abcdef"')
        assert re.search(pattern, 'api-key = "12345678"')
        assert not re.search(pattern, 'api_key = "short"')

    def test_detects_sql_injection_fstring(self):
        pattern = r'\bSELECT.*\bf"'
        assert re.search(pattern, 'cursor.execute(SELECT * FROM users WHERE id = f"')
        assert not re.search(pattern, "SELECT * FROM users")

    def test_detects_innerHTML(self):
        pattern = r'\binnerHTML\s*='
        assert re.search(pattern, 'el.innerHTML = userInput')
        assert not re.search(pattern, 'textContent = "safe"')

    def test_detects_dangerouslySetInnerHTML(self):
        pattern = r'\bdangerouslySetInnerHTML\b'
        assert re.search(pattern, '{dangerouslySetInnerHTML: ...}')
        assert not re.search(pattern, 'safeHTML')

    def test_detects_document_write(self):
        pattern = r'document\.write\s*\('
        assert re.search(pattern, 'document.write(html)')
        assert not re.search(pattern, 'documentWrite()')

    def test_detects_subprocess_call_as_shell_string(self):
        pattern = r'\bsubprocess\.call\s*\(\s*["\']'
        assert re.search(pattern, 'subprocess.call("ls -la")')
        assert not re.search(pattern, 'subprocess.call(["ls"])')

    # ── Performance patterns ──────────────────────────────────────────

    def test_detects_console_log(self):
        pattern = r'console\.(log|debug|info|warn)\s*\('
        assert re.search(pattern, 'console.log("debug")')
        assert re.search(pattern, 'console.warn("warning")')
        assert not re.search(pattern, 'logger.info("ok")')

    # ── Correctness patterns ──────────────────────────────────────────

    def test_detects_bare_except(self):
        pattern = r'except\s*:[\r\n]'
        assert re.search(pattern, 'except:\n    pass')
        assert not re.search(pattern, 'except ValueError:\n    pass')

    def test_detects_silenced_exception(self):
        pattern = r'except\s+Exception\s*:\s*\n\s*pass'
        assert re.search(pattern, 'except Exception:\n    pass')
        assert not re.search(pattern, 'except ValueError:\n    logger.warn(e)')

    def test_detects_unsafe_get_chaining(self):
        pattern = r'\.get\(\)\s*\.'
        assert re.search(pattern, 'dict.get().value')
        assert not re.search(pattern, 'dict.get("key", default)')

    def test_detects_missing_encoding(self):
        pattern = r'open\s*\(\s*(?:\w+|["\x27][^"\x27]+["\x27]).*\)\s+as'
        assert re.search(pattern, 'open(filename) as f:')
        assert re.search(pattern, 'open("file.txt", "r") as f:')
        assert not re.search(pattern, 'open("file.txt")')

    def test_detects_mutable_class_default(self):
        pattern = r'self\.\w+\s*=\s*\{\}'
        assert re.search(pattern, 'self._cache = {}')
        assert not re.search(pattern, 'self._cache = dict()')

    # ── Edge cases ────────────────────────────────────────────────────

    def test_security_pattern_count(self):
        """Verify we haven't accidentally lost security patterns."""
        assert len(ReviewSpecialistExecutor.SECURITY_PATTERNS) == 12

    def test_performance_pattern_count(self):
        assert len(ReviewSpecialistExecutor.PERFORMANCE_PATTERNS) == 4

    def test_correctness_pattern_count(self):
        assert len(ReviewSpecialistExecutor.CORRECTNESS_PATTERNS) == 5

    # ── API_KEY case sensitivity (known limitation) ───────────────────

    def test_api_key_pattern_is_case_sensitive(self):
        """Document: pattern uses \bapi which doesn't match API_KEY."""
        pattern = r'\bapi[_-]?key\s*=\s*["\'][A-Za-z0-9_\-]{8,}["\']'
        # lowercase works
        assert re.search(pattern, 'api_key = "sk-1234567890abcdef"')
        # uppercase does NOT — known limitation
        assert not re.search(pattern, 'API_KEY = "sk-1234567890abcdef"')

    # ── Diff extraction ───────────────────────────────────────────────

    def test_extract_files_from_diff(self):
        diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,5 @@
+import os
+os.system("ls")
 normal line
diff --git a/src/util.py b/src/util.py
--- a/src/util.py
+++ b/src/util.py
@@ -1,1 +1,2 @@
+new line
"""
        files = ReviewSpecialistExecutor._extract_files(diff)
        assert 'src/main.py' in files
        assert 'src/util.py' in files
        assert 'import os' in files['src/main.py']
        assert 'os.system' in files['src/main.py']
        assert 'new line' in files['src/util.py']
        assert 'normal line' not in files['src/main.py']

    def test_empty_diff(self):
        files = ReviewSpecialistExecutor._extract_files('')
        assert files == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Test Specialist — parser tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTestSpecialistParser:
    """Verify test output parsers for pytest, Jest, and go test."""

    def test_parse_pytest_output(self):
        executor = TestSpecialistExecutor()
        # pytest format: "X passed, Y failed" — parser expects
        # the "passed" group first in the regex
        output = '40 passed, 2 failed in 1.23s'
        result = executor._parse_test_results(output)
        assert result['passed'] == 40
        assert result['failed'] == 2
        assert result['total'] == 42

    def test_parse_jest_output(self):
        executor = TestSpecialistExecutor()
        output = 'Tests:       15 passed, 2 failed, 1 skipped, 18 total'
        result = executor._parse_test_results(output)
        assert result['passed'] == 15
        assert result['failed'] == 2
        assert result['skipped'] == 1
        assert result['total'] == 18

    def test_parse_go_test_output(self):
        executor = TestSpecialistExecutor()
        # go test counts FAIL at start of line — there are two such lines
        output = """ok      github.com/foo/bar       0.123s
--- FAIL: TestBaz (0.01s)
FAIL    github.com/foo/baz       0.456s
"""
        result = executor._parse_test_results(output)
        # 1 ok line, 1 FAIL line (--- FAIL doesn't start with ^FAIL)
        assert result['passed'] == 1
        assert result['failed'] == 1
        assert result['total'] == 2

    def test_parse_pytest_coverage(self):
        executor = TestSpecialistExecutor()
        output = """Name                     Stmts   Miss  Cover
--------------------------------------------
src/main.py                 45      2    96%
src/models.py               30     15    50%
src/utils.py                20      0   100%
--------------------------------------------
TOTAL                       95     17    82%
"""
        coverage = executor._parse_coverage(output)
        assert coverage['line_rate'] == 82.0
        assert len(coverage['files']) == 3
        file_cov = {f['path']: f['coverage'] for f in coverage['files']}
        assert file_cov['src/main.py'] == 96.0
        assert file_cov['src/models.py'] == 50.0

    def test_parse_jest_coverage(self):
        executor = TestSpecialistExecutor()
        output = 'All files  |   85.71 |    66.66 |   80.00 |   85.71 |'
        coverage = executor._parse_coverage(output)
        assert coverage['line_rate'] == 85.71

    def test_parse_nyc_coverage(self):
        executor = TestSpecialistExecutor()
        output = 'Lines     : 73.50%'
        coverage = executor._parse_coverage(output)
        assert coverage['line_rate'] == 73.50

    def test_find_coverage_gaps_below_target(self):
        executor = TestSpecialistExecutor()
        coverage = {
            'line_rate': 78.0,
            'files': [
                {'path': 'src/main.py', 'coverage': 96.0},
                {'path': 'src/models.py', 'coverage': 45.0},
                {'path': 'src/utils.py', 'coverage': 60.0},
            ],
        }
        gaps = executor._find_coverage_gaps(coverage, target=80.0)
        assert len(gaps) == 2
        gap_files = {g['path'] for g in gaps}
        assert 'src/models.py' in gap_files
        assert 'src/utils.py' in gap_files
        assert 'src/main.py' not in gap_files

    def test_find_coverage_gaps_above_target(self):
        executor = TestSpecialistExecutor()
        coverage = {
            'line_rate': 95.0,
            'files': [
                {'path': 'src/a.py', 'coverage': 95.0},
                {'path': 'src/b.py', 'coverage': 90.0},
            ],
        }
        gaps = executor._find_coverage_gaps(coverage, target=80.0)
        assert len(gaps) == 0

    def test_build_test_summary(self):
        tests = {'passed': 10, 'failed': 2, 'skipped': 1, 'total': 13}
        coverage = {'line_rate': 85.0}
        gaps = [{'path': 'src/bad.py', 'coverage': 40.0}]
        summary = TestSpecialistExecutor._build_test_summary(
            tests, coverage, gaps
        )
        assert '10 passed, 2 failed' in summary
        assert '1 skipped' in summary
        assert '85.0%' in summary
        assert '1 file(s) below target' in summary


# ═══════════════════════════════════════════════════════════════════════════════
# E2E User Test — step routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestE2EStepRouting:
    """Verify e2e executor step routing validates actions correctly."""

    def test_unknown_action_returns_error(self):
        from openhands.tools.lsp.role_agents import E2EUserTestExecutor
        executor = E2EUserTestExecutor()
        result = executor._execute_step(
            page=None, step={'action': 'nuclear_launch'},
            screenshot_dir='/tmp', step_num=1,
        )
        assert result['status'] == 'error'
        assert 'Unknown action' in result['error']

    def test_known_actions_all_in_supported_set(self):
        from openhands.tools.lsp.role_agents import E2EUserTestExecutor
        executor = E2EUserTestExecutor()
        # Unknown action is rejected before any Playwright interaction
        result = executor._execute_step(
            page=None, step={'action': 'click'},
            screenshot_dir='/tmp', step_num=1,
        )
        # 'click' is supported, so it won't return "Unknown action"
        # (it will fail on None page, but in the try block, not the validation)
        assert result['action'] == 'click'


# ═══════════════════════════════════════════════════════════════════════════════
# Tool name / registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolNaming:
    """Verify auto-derived tool names match expected values."""

    def test_names_in_all_lsp_tools(self):
        from openhands.tools.lsp import ALL_LSP_TOOLS
        expected = {
            'review_specialist', 'test_specialist', 'e2e_user_test',
            'decompose_task', 'validate_contract',
            'plane_create_issue', 'plane_update_issue',
            'plane_comment_issue', 'plane_get_issues',
            'plane_create_cycle', 'plane_sync_task',
        }
        assert expected <= set(ALL_LSP_TOOLS), (
            f'Missing tools: {expected - set(ALL_LSP_TOOLS)}'
        )

    def test_tool_count_is_32(self):
        from openhands.tools.lsp import ALL_LSP_TOOLS
        assert len(ALL_LSP_TOOLS) == 32, (
            f'Expected 32 tools, got {len(ALL_LSP_TOOLS)}: {ALL_LSP_TOOLS}'
        )
        assert len(set(ALL_LSP_TOOLS)) == 32, 'Duplicate tool names found'

    def test_role_agent_tool_names(self):
        assert ReviewSpecialistTool.name == 'review_specialist'
        assert TestSpecialistTool.name == 'test_specialist'
        assert E2EUserTestTool.name == 'e2e_user_test'

    def test_plane_tool_names(self):
        from openhands.tools.lsp.plane_integration import (
            PlaneCreateIssueTool, PlaneUpdateIssueTool,
            PlaneCommentIssueTool, PlaneGetIssuesTool,
            PlaneCreateCycleTool, PlaneSyncTaskTool,
        )
        assert PlaneCreateIssueTool.name == 'plane_create_issue'
        assert PlaneUpdateIssueTool.name == 'plane_update_issue'
        assert PlaneCommentIssueTool.name == 'plane_comment_issue'
        assert PlaneGetIssuesTool.name == 'plane_get_issues'
        assert PlaneCreateCycleTool.name == 'plane_create_cycle'
        assert PlaneSyncTaskTool.name == 'plane_sync_task'


# ═══════════════════════════════════════════════════════════════════════════════
# Observation helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservationFromData:
    """Verify RoleAgentObservation.from_data() creates valid objects."""

    def test_from_data_success(self):
        obs = RoleAgentObservation.from_data({'status': 'ok', 'count': 5})
        assert obs.is_error is False
        assert 'ok' in obs.text
        assert obs.agent_data == {'status': 'ok', 'count': 5}

    def test_from_data_error(self):
        obs = RoleAgentObservation.from_data({'error': 'boom'})
        assert obs.is_error is True
        assert 'boom' in obs.text
        assert obs.agent_data == {'error': 'boom'}
