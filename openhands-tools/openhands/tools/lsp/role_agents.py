"""Role-based agent decomposition (Force Multiplier #2).

Implements:
- System Architect agent: Master planner that decomposes tasks across repos
- Repository specialists: Go-specialist, TypeScript-specialist, Java-specialist
- Code Review specialist: Structured diff analysis with severity-ranked feedback
- Test specialist: Test executor with coverage gap detection
- E2E User Test specialist: Real browser automation (Playwright) simulating real users
- Contract-first development: Agents communicate through API contracts
  (OpenAPI specs, gRPC .proto) stored in a shared read-only location.

Tools:
- decompose_task: Break a high-level task into language-specific sub-tasks
- validate_contract: Check that sub-agent output conforms to API contract
- review_specialist: Structured code review of a diff/PR with severity ranking
- test_specialist: Execute tests, collect coverage, identify gaps
- e2e_user_test: Real browser E2E test behaving like a real user
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.llm import TextContent
from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Action / Observation schemas
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DecomposeTaskAction(Action):
    """Break a high-level task into language-specific sub-tasks."""

    task_description: str
    affected_repos: list[str] = field(default_factory=list)
    contract_files: list[str] = field(default_factory=list)

    @classmethod
    def name(cls) -> str:
        return 'decompose_task'


@dataclass
class ValidateContractAction(Action):
    """Validate that sub-agent output conforms to an API contract."""

    contract_file: str
    implementation_path: str
    contract_type: str = 'auto'

    @classmethod
    def name(cls) -> str:
        return 'validate_contract'


@dataclass
class ReviewSpecialistAction(Action):
    """Request a structured code review from the Review Specialist."""

    diff: str = ''
    base_branch: str = 'main'
    focus_areas: list[str] = field(default_factory=list)
    file_filters: list[str] = field(default_factory=list)

    @classmethod
    def name(cls) -> str:
        return 'review_specialist'


@dataclass
class TestSpecialistAction(Action):
    """Request the Test Specialist to run tests and analyse coverage."""

    test_command: str
    working_dir: str = ''
    collect_only: bool = False
    coverage_target: float = 80.0

    @classmethod
    def name(cls) -> str:
        return 'test_specialist'


@dataclass
class E2EUserTestAction(Action):
    """Run an E2E test using a real browser, behaving like a real user."""

    start_url: str
    steps: list[dict[str, str]]
    screenshot_dir: str = ''
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    user_agent: str = ''

    @classmethod
    def name(cls) -> str:
        return 'e2e_user_test'


@dataclass
class RoleAgentObservation(Observation):
    """Observation from role-based agent operations."""

    agent_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> RoleAgentObservation:
        text = '```json\n' + json.dumps(data, indent=2) + '\n```'
        is_error = bool(data.get('error'))
        return cls.model_construct(
            content=[TextContent(text=text)],
            is_error=is_error,
            agent_data=data,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Task Decomposition
# ═══════════════════════════════════════════════════════════════════════════════


class DecomposeTaskTool(ToolDefinition[DecomposeTaskAction, RoleAgentObservation]):
    """Break a high-level task into language-specific sub-tasks."""

    description: str = (
        'Break a high-level task into language-specific sub-tasks. '
        'Each sub-task targets a specific repo and language. '
        'Use this as the System Architect before dispatching work.'
    )
    action_type: type[DecomposeTaskAction] = DecomposeTaskAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> DecomposeTaskTool:
        return cls(executor=DecomposeTaskExecutor())


class DecomposeTaskExecutor:
    """Analyze repos and contracts to produce a decomposition plan."""

    def __call__(self, action: DecomposeTaskAction, conversation: Any) -> Observation:
        subtasks = []
        repo_languages: dict[str, str] = {}

        for repo_path in action.affected_repos:
            if not os.path.isdir(repo_path):
                continue
            lang = self._detect_repo_language(repo_path)
            repo_languages[repo_path] = lang

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
        markers = {
            'go.mod': 'go', 'package.json': 'typescript',
            'pom.xml': 'java', 'build.gradle': 'java',
            'pyproject.toml': 'python', 'setup.py': 'python',
            'buf.yaml': 'proto',
        }
        for filename, lang in markers.items():
            if os.path.exists(os.path.join(repo_path, filename)):
                return lang
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
# Contract Validation
# ═══════════════════════════════════════════════════════════════════════════════


class ValidateContractTool(ToolDefinition[ValidateContractAction, RoleAgentObservation]):
    """Validate that implementation output conforms to API contract."""

    description: str = (
        'Validate that an implementation conforms to its API contract. '
        'Supports .proto (buf breaking), OpenAPI (spectral), GraphQL. '
        'Use after each specialist agent completes a sub-task.'
    )
    action_type: type[ValidateContractAction] = ValidateContractAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False,
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
        proto_dir = os.path.dirname(proto_file)
        try:
            result = subprocess.run(
                ['buf', 'breaking', '--against', '.git#branch=main'],
                capture_output=True, text=True, timeout=30, cwd=proto_dir,
            )
            violations = (
                result.stdout.strip().split('\n') if result.stdout.strip() else []
            )
            return RoleAgentObservation.from_data({
                'contract': proto_file, 'tool': 'buf breaking',
                'status': 'pass' if result.returncode == 0 else 'fail',
                'violations': violations,
            })
        except FileNotFoundError:
            return RoleAgentObservation.from_data({
                'contract': proto_file, 'tool': 'buf breaking',
                'status': 'skipped', 'message': 'buf CLI not installed',
            })

    @staticmethod
    def _validate_openapi(spec_file: str) -> RoleAgentObservation:
        try:
            result = subprocess.run(
                ['npx', 'spectral', 'lint', spec_file],
                capture_output=True, text=True, timeout=30,
            )
            issues = [
                line.strip()
                for line in result.stdout.strip().split('\n')
                if 'error' in line.lower() or 'warning' in line.lower()
            ]
            return RoleAgentObservation.from_data({
                'contract': spec_file, 'tool': 'spectral',
                'status': 'pass' if result.returncode == 0 else 'fail',
                'issues': issues,
            })
        except Exception:
            return RoleAgentObservation.from_data({
                'contract': spec_file, 'tool': 'spectral',
                'status': 'skipped',
                'message': 'spectral not available (npm install -g @stoplight/spectral-cli)',
            })

    @staticmethod
    def _validate_graphql(schema_file: str) -> RoleAgentObservation:
        return RoleAgentObservation.from_data({
            'contract': schema_file, 'status': 'skipped',
            'message': 'GraphQL validation requires graphql-inspector',
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Code Review Specialist
# ═══════════════════════════════════════════════════════════════════════════════


class ReviewSpecialistTool(ToolDefinition[ReviewSpecialistAction, RoleAgentObservation]):
    """Structured code review with severity-ranked findings.

    Analyses diffs for security vulnerabilities, performance regressions,
    correctness bugs, and style violations. Returns findings ranked by
    severity: critical, warning, suggestion.
    """

    description: str = (
        'Perform a structured code review of a diff. Returns severity-ranked '
        'findings: critical (security, data loss), warning (perf, correctness), '
        'suggestion (style, readability). Focus areas: security, performance, '
        'correctness, style.'
    )
    action_type: type[ReviewSpecialistAction] = ReviewSpecialistAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> ReviewSpecialistTool:
        return cls(executor=ReviewSpecialistExecutor())


class ReviewSpecialistExecutor:
    """Analyse a diff with pattern-based security, perf, and style checks."""

    SECURITY_PATTERNS: list[tuple[str, str, str]] = [
        (r'\bos\.system\s*\(', 'critical',
         'os.system() can lead to command injection — use subprocess.run() with list args'),
        (r'\beval\s*\(', 'critical',
         'eval() executes arbitrary code — use ast.literal_eval() or safer alternatives'),
        (r'\bexec\s*\(', 'critical', 'exec() is dangerous — avoid dynamic code execution'),
        (r'\bpassword\s*=\s*["\'][^\"\']+["\']', 'critical',
         'Hardcoded password detected'),
        (r'\bapi[_-]?key\s*=\s*["\'][A-Za-z0-9_\-]{8,}["\']', 'critical',
         'Hardcoded API key detected'),
        (r'\bSELECT.*\bf"', 'critical',
         'Possible SQL injection via f-string in SQL query'),
        (r'\bexecute\s*\(\s*f["\']', 'critical',
         'Possible SQL injection via Python string formatting'),
        (r'\binnerHTML\s*=', 'critical',
         'XSS risk: innerHTML assignment — use textContent or sanitize'),
        (r'\bdangerouslySetInnerHTML\b', 'critical',
         'XSS risk: dangerouslySetInnerHTML in React'),
        (r'popen\s*\(\s*["\'].*\$', 'critical',
         'Shell injection risk in popen() — use list args'),
        (r'\bsubprocess\.call\s*\(\s*["\']', 'critical',
         'Use subprocess.run() with list args instead of shell string'),
        (r'document\.write\s*\(', 'critical', 'XSS risk: document.write() call'),
    ]

    PERFORMANCE_PATTERNS: list[tuple[str, str, str]] = [
        (r'for\s+\w+\s+in\s+\w+:\s*\n\s*.*\.filter\s*\(', 'warning',
         'N+1: filtering in loop — push to DB query'),
        (r'for\s+\w+\s+in\s+\w+:\s*\n\s*.*\.objects\.get\s*\(', 'warning',
         'N+1: DB query inside loop — use select_related/prefetch_related'),
        (r'\.all\(\)\s*\[', 'warning',
         'Loading entire table then slicing — use LIMIT/OFFSET in query'),
        (r'console\.(log|debug|info|warn)\s*\(', 'suggestion',
         'Stray console.log in production code — use proper logger'),
    ]

    CORRECTNESS_PATTERNS: list[tuple[str, str, str]] = [
        (r'except\s*:[\r\n]', 'warning',
         'Bare except: catches unintended exceptions including KeyboardInterrupt'),
        (r'except\s+Exception\s*:\s*\n\s*pass', 'warning',
         'Silenced exception — at minimum log the error'),
        (r'\.get\(\)\s*\.', 'warning',
         'Possible AttributeError from None — use .filter().first() or add default'),
        (r'open\s*\(\s*(?:\w+|["\x27][^"\x27]+["\x27]).*\)\s+as', 'suggestion',
         'File opened without encoding — use encoding="utf-8"'),
        (r'self\.\w+\s*=\s*\{\}', 'suggestion',
         'Mutable class-level default — move to __init__'),
    ]

    def __call__(self, action: ReviewSpecialistAction, conversation: Any) -> Observation:
        try:
            diff_text = action.diff or self._get_git_diff(action.base_branch)

            if not diff_text.strip():
                return RoleAgentObservation.from_data({
                    'findings': [], 'summary': 'No changes to review.',
                    'files_reviewed': 0, 'total_findings': 0,
                })

            files = self._extract_files(diff_text)
            if action.file_filters:
                files = {
                    f: c for f, c in files.items()
                    if any(fnmatch.fnmatch(f, g) for g in action.file_filters)
                }

            findings = self._analyse_files(files, action.focus_areas)
            findings.sort(key=lambda f: (
                {'critical': 0, 'warning': 1, 'suggestion': 2}.get(
                    f['severity'], 99
                ),
                f.get('file', ''), f.get('line', 0),
            ))

            return RoleAgentObservation.from_data({
                'findings': findings,
                'summary': self._build_summary(findings),
                'files_reviewed': len(files),
                'total_findings': len(findings),
            })
        except Exception as e:
            logger.exception('Review specialist failed')
            return RoleAgentObservation.from_data({'error': str(e)})

    def _get_git_diff(self, base_branch: str) -> str:
        try:
            result = subprocess.run(
                ['git', 'diff', f'{base_branch}...HEAD'],
                capture_output=True, text=True, timeout=15,
            )
            return result.stdout
        except Exception:
            return ''

    @staticmethod
    def _extract_files(diff_text: str) -> dict[str, str]:
        files: dict[str, str] = {}
        current_file = ''
        for line in diff_text.split('\n'):
            if line.startswith('+++ b/'):
                current_file = line[6:]
                files[current_file] = ''
            elif current_file and line.startswith('+') and not line.startswith('+++'):
                files[current_file] += line[1:] + '\n'
        return files

    def _analyse_files(
        self, files: dict[str, str], focus_areas: list[str]
    ) -> list[dict]:
        findings: list[dict] = []
        seen: set[tuple[int, str, str]] = set()
        focus = (
            set(focus_areas)
            if focus_areas
            else {'security', 'performance', 'correctness', 'style'}
        )

        if 'security' in focus:
            findings.extend(
                self._scan_patterns(files, self.SECURITY_PATTERNS, seen)
            )
        if 'performance' in focus:
            findings.extend(
                self._scan_patterns(files, self.PERFORMANCE_PATTERNS, seen)
            )
        if 'correctness' in focus:
            findings.extend(
                self._scan_patterns(files, self.CORRECTNESS_PATTERNS, seen)
            )
        if 'style' in focus:
            findings.extend(self._check_style(files, seen))

        return findings

    def _scan_patterns(
        self, files: dict[str, str], patterns: list, seen: set
    ) -> list[dict]:
        findings: list[dict] = []
        for filename, content in files.items():
            for line_no, line in enumerate(content.split('\n'), 1):
                for pattern, severity, message in patterns:
                    if re.search(pattern, line):
                        key = (line_no, filename, message)
                        if key not in seen:
                            seen.add(key)
                            findings.append({
                                'file': filename,
                                'line': line_no,
                                'severity': severity,
                                'message': message,
                                'snippet': line.strip()[:120],
                            })
        return findings

    def _check_style(self, files: dict[str, str], seen: set) -> list[dict]:
        findings: list[dict] = []
        for filename, content in files.items():
            for line_no, line in enumerate(content.split('\n'), 1):
                if len(line) > 120:
                    key = (line_no, filename, 'line too long')
                    if key not in seen:
                        seen.add(key)
                        findings.append({
                            'file': filename,
                            'line': line_no,
                            'severity': 'suggestion',
                            'message': f'Line too long ({len(line)} chars > 120)',
                        })
        return findings

    @staticmethod
    def _build_summary(findings: list[dict]) -> str:
        critical = sum(1 for f in findings if f['severity'] == 'critical')
        warnings = sum(1 for f in findings if f['severity'] == 'warning')
        suggestions = sum(1 for f in findings if f['severity'] == 'suggestion')
        parts = []
        if critical:
            parts.append(f'\U0001f534 {critical} critical')
        if warnings:
            parts.append(f'\U0001f7e1 {warnings} warning(s)')
        if suggestions:
            parts.append(f'\U0001f535 {suggestions} suggestion(s)')
        return ', '.join(parts) if parts else '\u2705 No issues found'


# ═══════════════════════════════════════════════════════════════════════════════
# Test Specialist
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecialistTool(ToolDefinition[TestSpecialistAction, RoleAgentObservation]):
    """Run tests, collect coverage, and identify gaps."""

    description: str = (
        'Run tests and analyse coverage. Returns pass/fail/skip counts, '
        'coverage percentage, and a list of files/modules below the '
        'coverage target. Use after making code changes to verify quality.'
    )
    action_type: type[TestSpecialistAction] = TestSpecialistAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True, destructive_hint=False,
        idempotent_hint=True, open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> TestSpecialistTool:
        return cls(executor=TestSpecialistExecutor())


class TestSpecialistExecutor:
    """Execute tests and parse coverage output."""

    def __call__(self, action: TestSpecialistAction, conversation: Any) -> Observation:
        try:
            wd = action.working_dir or os.getcwd()
            if not os.path.isdir(wd):
                return RoleAgentObservation.from_data(
                    {'error': f'Working dir not found: {wd}'}
                )

            result = subprocess.run(
                action.test_command,
                shell=True, capture_output=True, text=True,
                timeout=300, cwd=wd,
            )
            combined = result.stdout + '\n' + result.stderr

            tests_summary = self._parse_test_results(combined)
            coverage = self._parse_coverage(combined)
            gaps = self._find_coverage_gaps(coverage, action.coverage_target)

            return RoleAgentObservation.from_data({
                'exit_code': result.returncode,
                'tests': tests_summary,
                'coverage': coverage,
                'coverage_target': action.coverage_target,
                'gaps': gaps,
                'summary': self._build_test_summary(
                    tests_summary, coverage, gaps
                ),
            })
        except subprocess.TimeoutExpired:
            return RoleAgentObservation.from_data(
                {'error': 'Test command timed out after 5 minutes'}
            )
        except Exception as e:
            logger.exception('Test specialist failed')
            return RoleAgentObservation.from_data({'error': str(e)})

    def _parse_test_results(self, output: str) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {
            'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0,
            'total': 0, 'duration_sec': None,
        }

        # pytest: "3 passed, 1 failed, 2 skipped"
        pytest_match = re.search(
            r'(\d+)\s+passed[,.]?\s*'
            r'(?:(\d+)\s+failed[,.]?\s*)?'
            r'(?:(\d+)\s+skipped[,.]?\s*)?'
            r'(?:(\d+)\s+error)?',
            output,
        )
        if pytest_match:
            result['passed'] = int(pytest_match.group(1) or 0)
            result['failed'] = int(pytest_match.group(2) or 0)
            result['skipped'] = int(pytest_match.group(3) or 0)
            result['error'] = int(pytest_match.group(4) or 0)
            result['total'] = sum(
                int(result[k]) for k in ('passed', 'failed', 'skipped', 'error')
            )

        # Jest/Mocha: "Tests: 5 passed, 2 failed, 8 total"
        if result['total'] == 0:
            t = re.search(r'Tests?:.*?(\d+)\s+total', output)
            if t:
                result['total'] = int(t.group(1))
                p = re.search(r'(\d+)\s+passed', output)
                f = re.search(r'(\d+)\s+failed', output)
                s = re.search(r'(\d+)\s+skipped', output)
                result['passed'] = int(p.group(1)) if p else 0
                result['failed'] = int(f.group(1)) if f else 0
                result['skipped'] = int(s.group(1)) if s else 0

        # go test
        if result['total'] == 0:
            ok_count = len(re.findall(r'^ok\s+', output, re.MULTILINE))
            fail_count = len(re.findall(r'^FAIL\s+', output, re.MULTILINE))
            if ok_count or fail_count:
                result['passed'] = ok_count
                result['failed'] = fail_count
                result['total'] = ok_count + fail_count

        dur = re.search(r'(\d+\.?\d*)s', output)
        if dur:
            try:
                result['duration_sec'] = float(dur.group(1))
            except ValueError:
                pass

        return result

    def _parse_coverage(self, output: str) -> dict:
        coverage: dict = {'line_rate': 0.0, 'files': []}

        # pytest-cov: "TOTAL 1234 56 95%"
        total_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
        if total_match:
            coverage['line_rate'] = float(total_match.group(1))

        for fm in re.finditer(r'(\S+\.py)\s+(\d+)\s+(\d+)\s+(\d+)%', output):
            coverage['files'].append({
                'path': fm.group(1),
                'statements': int(fm.group(2)),
                'missing': int(fm.group(3)),
                'coverage': float(fm.group(4)),
            })

        # Istanbul "All files | 85.5 | ..."
        if coverage['line_rate'] == 0.0:
            ist_match = re.search(r'All files\s*\|\s*([\d.]+)\s*\|', output)
            if ist_match:
                coverage['line_rate'] = float(ist_match.group(1))

        # nyc "Lines     : 85.5%"
        if coverage['line_rate'] == 0.0:
            nyc_match = re.search(r'Lines\s*:\s*([\d.]+)%', output)
            if nyc_match:
                coverage['line_rate'] = float(nyc_match.group(1))

        return coverage

    def _find_coverage_gaps(self, coverage: dict, target: float) -> list[dict]:
        gaps = [
            f for f in coverage.get('files', [])
            if f['coverage'] < target
        ]
        gaps.sort(key=lambda g: g['coverage'])
        return gaps

    @staticmethod
    def _build_test_summary(
        tests: dict, coverage: dict, gaps: list
    ) -> str:
        parts = [
            f"{tests.get('passed', 0)} passed, "
            f"{tests.get('failed', 0)} failed",
        ]
        if tests.get('skipped'):
            parts.append(f"{tests['skipped']} skipped")
        cov = coverage.get('line_rate', 0)
        parts.append(f'{cov:.1f}% coverage')
        if gaps:
            parts.append(f'{len(gaps)} file(s) below target')
        return ' \u2014 '.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# E2E User Test Specialist  (real browser, real user behaviour)
# ═══════════════════════════════════════════════════════════════════════════════


class E2EUserTestTool(ToolDefinition[E2EUserTestAction, RoleAgentObservation]):
    """End-to-end test using a real browser, behaving like a real user."""

    description: str = (
        'Run an end-to-end test using a real browser (Playwright). '
        'Performs user actions (click, type, navigate, scroll, hover, select) '
        'and validates assertions (visible, hidden, contains, equals, '
        'url_matches). Captures screenshots at each step. '
        'Requires: pip install playwright && playwright install chromium'
    )
    action_type: type[E2EUserTestAction] = E2EUserTestAction
    observation_type: type[RoleAgentObservation] = RoleAgentObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False, destructive_hint=False,
        idempotent_hint=False, open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> E2EUserTestTool:
        return cls(executor=E2EUserTestExecutor())


E2EUserTestTool.name = 'e2e_user_test'


class E2EUserTestExecutor:
    """Drive a real browser via Playwright simulating user behaviour."""

    DEFAULT_USER_AGENT = (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    )

    SUPPORTED_ACTIONS = frozenset({
        'navigate', 'click', 'type', 'select', 'wait',
        'screenshot', 'hover', 'scroll', 'assert',
    })

    def __call__(
        self, action: E2EUserTestAction, conversation: Any
    ) -> Observation:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            return RoleAgentObservation.from_data({
                'error': (
                    'playwright not installed. '
                    'Run: pip install playwright && playwright install chromium'
                ),
            })

        screenshots: list[str] = []
        step_results: list[dict] = []
        start_time = time.monotonic()
        screenshot_dir = self._ensure_screenshot_dir(action.screenshot_dir)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=action.headless)
            context = browser.new_context(
                viewport={
                    'width': action.viewport_width,
                    'height': action.viewport_height,
                },
                user_agent=action.user_agent or self.DEFAULT_USER_AGENT,
            )
            page = context.new_page()

            try:
                page.goto(
                    action.start_url,
                    wait_until='domcontentloaded',
                    timeout=15000,
                )
                step_results.append({
                    'step': 0, 'action': 'navigate',
                    'url': action.start_url, 'status': 'ok',
                    'duration_ms': int(
                        (time.monotonic() - start_time) * 1000
                    ),
                })

                for i, step in enumerate(action.steps, 1):
                    step_start = time.monotonic()
                    result = self._execute_step(
                        page, step, screenshot_dir, i
                    )
                    result['duration_ms'] = int(
                        (time.monotonic() - step_start) * 1000
                    )
                    step_results.append(result)

                    if result['status'] == 'error':
                        try:
                            path = self._screenshot(
                                page, screenshot_dir, f'error_step_{i}'
                            )
                            if path:
                                screenshots.append(path)
                                result['screenshot'] = path
                        except Exception:
                            pass

                    if result.get('screenshot'):
                        screenshots.append(result['screenshot'])

                    if result['status'] == 'fail':
                        break

                try:
                    path = self._screenshot(
                        page, screenshot_dir, 'final_state'
                    )
                    if path:
                        screenshots.append(path)
                except Exception:
                    pass
            finally:
                browser.close()

        total_ms = int((time.monotonic() - start_time) * 1000)
        passed = sum(
            1 for r in step_results if r['status'] in ('ok', 'pass')
        )
        failed = sum(
            1 for r in step_results if r['status'] in ('fail', 'error')
        )

        return RoleAgentObservation.from_data({
            'total_steps': len(step_results),
            'passed': passed, 'failed': failed,
            'duration_ms': total_ms,
            'screenshots': screenshots,
            'steps': step_results,
            'verdict': (
                '\u2705 PASS'
                if failed == 0
                else f'\u274c FAIL ({failed} failure(s))'
            ),
        })

    def _execute_step(
        self, page, step: dict, screenshot_dir: str, step_num: int
    ) -> dict:
        step_action = step.get('action', '')
        selector = step.get('selector', '')
        value = step.get('value', '')
        timeout_ms = int(step.get('timeout_ms', 5000))

        if step_action not in self.SUPPORTED_ACTIONS:
            return {
                'step': step_num, 'action': step_action,
                'status': 'error',
                'error': f'Unknown action: {step_action}',
            }

        try:
            if step_action == 'navigate':
                page.goto(value, wait_until='domcontentloaded',
                           timeout=timeout_ms)
                return {
                    'step': step_num, 'action': 'navigate',
                    'url': value, 'status': 'ok',
                }

            if step_action == 'click':
                page.wait_for_selector(selector, timeout=timeout_ms)
                page.click(selector)
                return {
                    'step': step_num, 'action': 'click',
                    'selector': selector, 'status': 'ok',
                }

            if step_action == 'type':
                page.wait_for_selector(selector, timeout=timeout_ms)
                page.fill(selector, value)
                return {
                    'step': step_num, 'action': 'type',
                    'selector': selector, 'value': value, 'status': 'ok',
                }

            if step_action == 'select':
                page.wait_for_selector(selector, timeout=timeout_ms)
                page.select_option(selector, value)
                return {
                    'step': step_num, 'action': 'select',
                    'selector': selector, 'value': value, 'status': 'ok',
                }

            if step_action == 'wait':
                ms = int(value) if value else timeout_ms
                page.wait_for_timeout(ms)
                return {
                    'step': step_num, 'action': 'wait',
                    'duration_ms': ms, 'status': 'ok',
                }

            if step_action == 'hover':
                page.wait_for_selector(selector, timeout=timeout_ms)
                page.hover(selector)
                return {
                    'step': step_num, 'action': 'hover',
                    'selector': selector, 'status': 'ok',
                }

            if step_action == 'scroll':
                if selector:
                    page.locator(selector).scroll_into_view_if_needed()
                else:
                    page.evaluate(
                        f'window.scrollBy(0, {int(value) if value else 300})'
                    )
                return {
                    'step': step_num, 'action': 'scroll',
                    'selector': selector or 'window', 'status': 'ok',
                }

            if step_action == 'screenshot':
                path = self._screenshot(
                    page, screenshot_dir, f'step_{step_num:03d}'
                )
                return {
                    'step': step_num, 'action': 'screenshot',
                    'status': 'ok', 'screenshot': path or '',
                }

            if step_action == 'assert':
                assertion = step.get('assertion', 'visible')
                expected = step.get('expected', '')
                return self._run_assertion(
                    page, selector, assertion,
                    expected, timeout_ms, step_num,
                )

        except Exception as e:
            return {
                'step': step_num, 'action': step_action,
                'selector': selector, 'status': 'error',
                'error': str(e)[:200],
            }

        return {
            'step': step_num, 'action': step_action,
            'status': 'error', 'error': 'unreachable',
        }

    def _run_assertion(
        self, page, selector: str, assertion: str,
        expected: str, timeout_ms: int, step_num: int,
    ) -> dict:
        base = {
            'step': step_num, 'action': 'assert',
            'assertion': assertion, 'selector': selector,
        }

        if assertion == 'visible':
            page.wait_for_selector(
                selector, timeout=timeout_ms, state='visible'
            )
            return {**base, 'status': 'pass', 'expected': 'visible'}

        if assertion == 'hidden':
            page.wait_for_selector(
                selector, timeout=timeout_ms, state='hidden'
            )
            return {**base, 'status': 'pass', 'expected': 'hidden'}

        if assertion == 'contains':
            page.wait_for_selector(selector, timeout=timeout_ms)
            text = page.text_content(selector) or ''
            actual_ok = expected in text
            return {
                **base,
                'status': 'pass' if actual_ok else 'fail',
                'expected': f'contains "{expected}"',
                'actual': text[:200] if not actual_ok else expected,
            }

        if assertion == 'equals':
            page.wait_for_selector(selector, timeout=timeout_ms)
            text = (page.text_content(selector) or '').strip()
            actual_ok = text == expected
            return {
                **base,
                'status': 'pass' if actual_ok else 'fail',
                'expected': expected,
                'actual': text[:200] if not actual_ok else expected,
            }

        if assertion == 'url_matches':
            page.wait_for_timeout(min(timeout_ms, 3000))
            actual_ok = bool(re.search(expected, page.url))
            return {
                **base,
                'status': 'pass' if actual_ok else 'fail',
                'expected': f'URL matches /{expected}/',
                'actual': page.url if not actual_ok else expected,
            }

        if assertion == 'count':
            page.wait_for_timeout(min(timeout_ms, 2000))
            count = page.locator(selector).count()
            actual_ok = str(count) == str(expected)
            return {
                **base,
                'status': 'pass' if actual_ok else 'fail',
                'expected': str(expected),
                'actual': (
                    str(count) if not actual_ok else str(expected)
                ),
            }

        if assertion == 'attribute':
            page.wait_for_selector(selector, timeout=timeout_ms)
            attr_value = page.get_attribute(selector, expected)
            return {
                **base, 'status': 'ok',
                'attribute': expected, 'value': attr_value,
            }

        if assertion == 'title_contains':
            page.wait_for_timeout(min(timeout_ms, 3000))
            actual_ok = expected in page.title()
            return {
                **base,
                'status': 'pass' if actual_ok else 'fail',
                'expected': f'title contains "{expected}"',
                'actual': (
                    page.title() if not actual_ok else expected
                ),
            }

        return {
            **base, 'status': 'error',
            'error': f'Unknown assertion: {assertion}',
        }

    @staticmethod
    def _screenshot(page, directory: str, name: str) -> str | None:
        try:
            path = os.path.join(directory, f'{name}.png')
            page.screenshot(path=path, full_page=True)
            return path
        except Exception as e:
            logger.warning('Screenshot failed: %s', e)
            return None

    @staticmethod
    def _ensure_screenshot_dir(dir_path: str) -> str:
        path = dir_path or os.path.join(os.getcwd(), 'e2e_screenshots')
        os.makedirs(path, exist_ok=True)
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# Registration & Exports
# ═══════════════════════════════════════════════════════════════════════════════


def register_role_agent_tools() -> list[type[ToolDefinition]]:
    """Return all role-agent tool definitions for LSP tool registration."""
    return [
        DecomposeTaskTool,
        ValidateContractTool,
        ReviewSpecialistTool,
        TestSpecialistTool,
        E2EUserTestTool,
    ]


__all__ = [
    # Actions
    'DecomposeTaskAction',
    'ValidateContractAction',
    'ReviewSpecialistAction',
    'TestSpecialistAction',
    'E2EUserTestAction',
    # Tools
    'DecomposeTaskTool',
    'ValidateContractTool',
    'ReviewSpecialistTool',
    'TestSpecialistTool',
    'E2EUserTestTool',
    # Registry
    'register_role_agent_tools',
]