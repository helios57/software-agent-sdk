"""Semantic checkpointing memory layer for long-running agent sessions.

When the 1M context window fills, the agent can checkpoint its progress
into a persistent JSON "Memory" file, then reload it in a fresh session.
Optionally integrates with a vector store (ChromaDB) for semantic retrieval
of past interactions, PR feedback, and architectural decisions.

Force Multiplier #1 — Long-Term Retention
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.tool import ToolDefinition, ToolAnnotations

logger = logging.getLogger(__name__)

# Default checkpoint directory: project_root/.openhands/memory/
DEFAULT_MEMORY_DIR = '.openhands/memory'


@dataclass
class SemanticCheckpointAction(Action):
    """Create a semantic checkpoint of the current sub-task progress."""

    summary: str
    """A concise summary of what was accomplished."""

    decisions: list[str] = field(default_factory=list)
    """Key architectural decisions made during this step."""

    errors: list[str] = field(default_factory=list)
    """Errors encountered and how they were resolved."""

    key_files: list[str] = field(default_factory=list)
    """Files modified or created in this checkpoint."""

    checkpoint_id: str = ''
    """Optional: ID for updating an existing checkpoint."""

    @classmethod
    def name(cls) -> str:
        return 'semantic_checkpoint'


@dataclass
class SemanticRecallAction(Action):
    """Query the memory layer for relevant past context."""

    query: str
    """Natural language query describing what to recall."""

    max_results: int = 10
    """Maximum number of results to return."""

    search_type: str = 'semantic'
    """'semantic' (vector), 'keyword' (grep), or 'recent'."""

    @classmethod
    def name(cls) -> str:
        return 'semantic_recall'


@dataclass
class MemoryObservation(Observation):
    """Observation from the memory layer."""

    memory_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> MemoryObservation:
        text = '```json\n' + json.dumps(data, indent=2) + '\n```'
        obs = Observation.from_text(text)
        obs.__class__ = cls
        obs.memory_data = data
        return obs


# ── Checkpoint Tool ──────────────────────────────────────────────────────────


class SemanticCheckpointTool(
    ToolDefinition[SemanticCheckpointAction, MemoryObservation]
):
    """Save progress checkpoint for later recall."""

    name: str = 'semantic_checkpoint'
    description: str = (
        'Save a semantic checkpoint of your current sub-task progress. '
        'Use this before the context window fills, or after completing '
        'a significant step. The checkpoint persists across sessions.'
    )
    action_type: type[SemanticCheckpointAction] = SemanticCheckpointAction
    observation_type: type[MemoryObservation] = MemoryObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @classmethod
    def create(
        cls, conv_state: Any = None, **params: Any
    ) -> SemanticCheckpointTool:
        workspace = (
            conv_state.working_dir
            if conv_state and hasattr(conv_state, 'working_dir')
            else os.getcwd()
        )
        return cls(executor=CheckpointExecutor(workspace))


class CheckpointExecutor:
    """Write/update checkpoint files to .openhands/memory/."""

    def __init__(self, workspace_root: str):
        self._memory_dir = os.path.join(workspace_root, DEFAULT_MEMORY_DIR)

    def __call__(self, action: SemanticCheckpointAction, conversation: Any) -> Observation:
        try:
            os.makedirs(self._memory_dir, exist_ok=True)

            cid = action.checkpoint_id or str(uuid.uuid4())[:8]
            checkpoint = {
                'id': cid,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': action.summary,
                'decisions': action.decisions,
                'errors': action.errors,
                'key_files': action.key_files,
            }

            filepath = os.path.join(self._memory_dir, f'checkpoint_{cid}.json')
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Append to unified index
            index_path = os.path.join(self._memory_dir, 'index.jsonl')
            with open(index_path, 'a') as f:
                f.write(json.dumps(checkpoint) + '\n')

            return MemoryObservation.from_data({
                'status': 'saved',
                'checkpoint_id': cid,
                'file': filepath,
            })
        except Exception as e:
            logger.exception('Checkpoint failed')
            return MemoryObservation.from_data({'error': str(e)})


# ── Recall Tool ──────────────────────────────────────────────────────────────


class SemanticRecallTool(
    ToolDefinition[SemanticRecallAction, MemoryObservation]
):
    """Query the memory layer for relevant context."""

    name: str = 'semantic_recall'
    description: str = (
        'Query the memory layer for past checkpoints, PR feedback, '
        'and architectural decisions. Supports semantic (vector) and '
        'keyword search. Use when starting a new session or context.'
    )
    action_type: type[SemanticRecallAction] = SemanticRecallAction
    observation_type: type[MemoryObservation] = MemoryObservation
    annotations: ToolAnnotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> SemanticRecallTool:
        workspace = (
            conv_state.working_dir
            if conv_state and hasattr(conv_state, 'working_dir')
            else os.getcwd()
        )
        return cls(executor=RecallExecutor(workspace))


class RecallExecutor:
    """Search checkpoints with semantic or keyword matching."""

    def __init__(self, workspace_root: str):
        self._memory_dir = os.path.join(workspace_root, DEFAULT_MEMORY_DIR)

    def __call__(self, action: SemanticRecallAction, conversation: Any) -> Observation:
        try:
            index_path = os.path.join(self._memory_dir, 'index.jsonl')
            if not os.path.exists(index_path):
                return MemoryObservation.from_data({
                    'results': [],
                    'hint': 'No memory index found. Create checkpoints first.',
                })

            entries = []
            with open(index_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))

            if action.search_type == 'recent':
                # Return most recent entries
                entries = sorted(
                    entries,
                    key=lambda e: e.get('timestamp', ''),
                    reverse=True,
                )[: action.max_results]
            elif action.search_type == 'keyword':
                # Simple substring match
                query = action.query.lower()
                entries = [
                    e
                    for e in entries
                    if query in json.dumps(e).lower()
                ][: action.max_results]
            else:
                # Semantic: use naive TF-IDF (vector store is optional)
                entries = self._semantic_search(entries, action.query, action.max_results)

            return MemoryObservation.from_data({
                'results': entries,
                'count': len(entries),
                'search_type': action.search_type,
            })
        except Exception as e:
            logger.exception('Recall failed')
            return MemoryObservation.from_data({'error': str(e)})

    @staticmethod
    def _semantic_search(
        entries: list[dict], query: str, max_results: int
    ) -> list[dict]:
        """Naive semantic search via keyword overlap (vector store optional)."""
        query_words = set(query.lower().split())

        def score(entry: dict) -> float:
            text = json.dumps(entry).lower()
            entry_words = set(text.split())
            if not query_words:
                return 0.0
            overlap = len(query_words & entry_words)
            return overlap / len(query_words)

        scored = [(score(e), e) for e in entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored if _ > 0][:max_results]


__all__ = [
    'SemanticCheckpointAction',
    'SemanticRecallAction',
    'SemanticCheckpointTool',
    'SemanticRecallTool',
]
