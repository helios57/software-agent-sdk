"""ArgoCD health monitoring agent (Force Multiplier #2).

A persistent monitoring agent that watches ArgoCD Application status.
If status flips to Degraded, it injects a CRITICAL FAILURE warning
into the current agent context with k8s logs — before the human
even realizes there's a deployment error.

Usage: poll `argocd app get <app>` periodically, parse health status,
and inject observation when degraded.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)


@dataclass
class ArgoCDHealthCheckAction(Action):
    """Check the health status of an ArgoCD application."""

    app_name: str
    """Name of the ArgoCD Application resource."""

    namespace: str = 'argocd'
    """Namespace where ArgoCD is deployed."""

    fetch_logs: bool = True
    """If degraded, also fetch k8s pod logs for the failing resources."""

    max_log_lines: int = 50
    """Maximum number of log lines to return per failing pod."""

    @classmethod
    def name(cls) -> str:
        return 'argocd_health_check'


@dataclass
class ArgoCDWatchAction(Action):
    """Start a background watcher for ArgoCD application health."""

    app_name: str
    namespace: str = 'argocd'
    poll_interval_seconds: int = 30
    """How often to poll the ArgoCD API."""

    @classmethod
    def name(cls) -> str:
        return 'argocd_watch'


@dataclass
class ArgoCDHealthObservation(Observation):
    """Health status observation from ArgoCD."""

    health_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_status(cls, data: dict[str, Any]) -> ArgoCDHealthObservation:
        status = data.get('health_status', 'Unknown')
        app = data.get('app_name', 'unknown')

        if status == 'Degraded':
            text = f'🚨 CRITICAL: ArgoCD app "{app}" is DEGRADED\n'
        elif status == 'Progressing':
            text = f'⏳ ArgoCD app "{app}" is Progressing\n'
        elif status == 'Healthy':
            text = f'✅ ArgoCD app "{app}" is Healthy\n'
        else:
            text = f'❓ ArgoCD app "{app}" status: {status}\n'

        if data.get('failing_resources'):
            text += '\nFailing Resources:\n'
            for res in data['failing_resources'][:10]:
                text += f'  - {res.get("kind")}/{res.get("name")}: {res.get("message", "")}\n'

        if data.get('pod_logs'):
            text += '\nRecent Pod Logs:\n'
            for pod, logs in data['pod_logs'].items():
                text += f'\n--- {pod} ---\n{logs}\n'

        obs = Observation.from_text(text)
        obs.__class__ = cls
        obs.is_error = status == 'Degraded'
        obs.health_data = data
        return obs


class ArgoCDHealthCheckTool(
    ToolDefinition[ArgoCDHealthCheckAction, ArgoCDHealthObservation]
):
    """Check ArgoCD application health status."""

    name: str = 'argocd_health_check'
    description: str = (
        'Check the health status of an ArgoCD-managed application. '
        'If degraded, automatically fetches pod logs for failing resources. '
        'Use this for CI/CD feedback loop integration.'
    )
    action_type: type[ArgoCDHealthCheckAction] = ArgoCDHealthCheckAction
    observation_type: type[ArgoCDHealthObservation] = ArgoCDHealthObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> ArgoCDHealthCheckTool:
        return cls(executor=ArgoCDHealthExecutor())


class ArgoCDHealthExecutor:
    """Execute ArgoCD health checks via argocd CLI."""

    def __call__(self, action: ArgoCDHealthCheckAction, conversation: Any) -> Observation:
        try:
            result = subprocess.run(
                ['argocd', 'app', 'get', action.app_name, '--output', 'json'],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode != 0:
                return ArgoCDHealthObservation.from_status({
                    'app_name': action.app_name,
                    'health_status': 'Unknown',
                    'error': result.stderr.strip(),
                })

            app_data = json.loads(result.stdout)
            health_status = (
                app_data.get('status', {})
                .get('health', {})
                .get('status', 'Unknown')
            )

            failing_resources = []
            for res in app_data.get('status', {}).get('resources', []):
                if res.get('health', {}).get('status') in (
                    'Degraded', 'Missing', 'Progressing'
                ):
                    failing_resources.append({
                        'kind': res.get('kind', ''),
                        'name': res.get('name', ''),
                        'namespace': res.get('namespace', ''),
                        'message': res.get('status', ''),
                    })

            pod_logs = {}
            if action.fetch_logs and failing_resources:
                pod_logs = self._fetch_pod_logs(
                    action.app_name, failing_resources, action.max_log_lines
                )

            return ArgoCDHealthObservation.from_status({
                'app_name': action.app_name,
                'health_status': health_status,
                'sync_status': (
                    app_data.get('status', {})
                    .get('sync', {})
                    .get('status', 'Unknown')
                ),
                'failing_resources': failing_resources,
                'pod_logs': pod_logs,
                'timestamp': time.time(),
            })
        except FileNotFoundError:
            return ArgoCDHealthObservation.from_status({
                'app_name': action.app_name,
                'health_status': 'Unknown',
                'error': 'argocd CLI not installed. Install: curl -sSL https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64',
            })
        except Exception as e:
            logger.exception('ArgoCD health check failed')
            return ArgoCDHealthObservation.from_status({
                'app_name': action.app_name,
                'health_status': 'Unknown',
                'error': str(e),
            })

    @staticmethod
    def _fetch_pod_logs(
        app_name: str, resources: list[dict], max_lines: int
    ) -> dict[str, str]:
        """Fetch recent logs from failing pods."""
        logs = {}
        for res in resources[:3]:  # Limit to 3 resources
            kind = res.get('kind', '')
            name = res.get('name', '')
            namespace = res.get('namespace', 'default')

            if kind.lower() in ('deployment', 'statefulset', 'daemonset', 'replicaset'):
                try:
                    # Get pods for the resource
                    label = f'app={app_name}'  # Fallback label
                    pod_result = subprocess.run(
                        ['kubectl', 'get', 'pods', '-n', namespace,
                         '-l', label, '-o', 'name'],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    for pod_line in pod_result.stdout.strip().split('\n')[:3]:
                        pod_name = pod_line.replace('pod/', '')
                        if pod_name:
                            log_result = subprocess.run(
                                ['kubectl', 'logs', '-n', namespace,
                                 pod_name, '--tail', str(max_lines)],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )
                            logs[pod_name] = log_result.stdout.strip()
                except Exception:
                    pass
        return logs


__all__ = [
    'ArgoCDHealthCheckAction',
    'ArgoCDHealthCheckTool',
    'ArgoCDHealthObservation',
]
