# ruff: noqa: T201
"""Validate CrewAI knowledge/RAG with XPU embeddings and XPU-backed Ollama.

This manual E2E test combines both independently validated paths:

* CrewAI -> Sentence Transformers -> PyTorch XPU for knowledge ingestion and query
* CrewAI -> existing Ollama service -> Intel GPU for answer generation

The synthetic facts are present only in the knowledge source, not in the task or
expected-output prompt. The test fails unless CrewAI retrieves those facts, all
observed embedding model execution uses XPU tensors, and Ollama reports nonzero
VRAM in memory after the final answer.

Example:
    export OLLAMA_HOST="http://localhost:11434"
    export OLLAMA_MODEL="qwen2.5:0.5b"
    export SENTENCE_TRANSFORMER_MODEL="all-MiniLM-L6-v2"
    python scripts/xpu_rag_ollama_e2e.py
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import httpx


KNOWLEDGE_MARKER = "ZEPHYR-ACCESS-POLICY"
PRIVATE_POLICY = (
    "Document ZEPHYR-ACCESS-POLICY. "
    "The authorization code is VIOLET-LANTERN. "
    "Temporary contractor credentials expire after 47 hours. "
    "Final approval belongs to Dr. Nia Quill. "
    "These rules supersede all earlier drafts."
)
REQUIRED_FACTS = ("VIOLET-LANTERN", "47 hours", "Dr. Nia Quill")


class E2ETestFailure(RuntimeError):
    """Raised when the combined RAG/XPU E2E test fails."""


@dataclass(frozen=True)
class EmbeddingCallEvidence:
    """Device evidence collected for one embedding-function call."""

    phase: str
    input_count: int
    parameter_devices: tuple[str, ...]
    buffer_devices: tuple[str, ...]
    forward_tensor_devices: tuple[str, ...]
    xpu_memory_allocated_bytes: int


@dataclass(frozen=True)
class ModelInMemory:
    """Ollama model in-memory evidence."""

    name: str
    size_bytes: int
    size_vram_bytes: int

    @property
    def vram_percent(self) -> float:
        if self.size_bytes <= 0:
            return 0.0
        return self.size_vram_bytes / self.size_bytes * 100.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E2ETestFailure(message)


def _tensor_devices(value: Any, torch: Any) -> set[str]:
    """Collect tensor device names recursively from model inputs and outputs."""
    if isinstance(value, torch.Tensor):
        return {str(value.device)}
    if isinstance(value, Mapping):
        devices: set[str] = set()
        for item in value.values():
            devices.update(_tensor_devices(item, torch))
        return devices
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        devices = set()
        for item in value:
            devices.update(_tensor_devices(item, torch))
        return devices
    return set()


def _api_root(base_url: str) -> str:
    """Return the native Ollama API root, removing a trailing /v1 if present."""
    parsed = urlsplit(base_url.strip())
    _require(parsed.scheme in {"http", "https"}, "OLLAMA_HOST must use HTTP or HTTPS")
    _require(bool(parsed.netloc), "OLLAMA_HOST must include a host and port")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")


def _model_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for model in payload.get("models", []):
        if not isinstance(model, dict):
            continue
        for key in ("name", "model"):
            value = model.get(key)
            if isinstance(value, str):
                names.add(value)
    return names


def _same_model(candidate: str, requested: str) -> bool:
    return candidate == requested or candidate.removesuffix(":latest") == requested


def _get_json(client: httpx.Client, path: str) -> dict[str, Any]:
    try:
        response = client.get(path)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise E2ETestFailure(f"Ollama request failed for {path}: {error}") from error
    _require(isinstance(payload, dict), f"Ollama returned invalid JSON for {path}")
    return payload


def _ollama_initial_checks(client: httpx.Client, model_name: str) -> None:
    installed = _model_names(_get_json(client, "/api/tags"))
    _require(
        any(_same_model(candidate, model_name) for candidate in installed),
        f"Model {model_name!r} is not installed. Available models: {sorted(installed)}",
    )


def _model_in_memory(client: httpx.Client, model_name: str) -> ModelInMemory:
    running = _get_json(client, "/api/ps")
    for model in running.get("models", []):
        if not isinstance(model, dict):
            continue
        candidate = model.get("name") or model.get("model")
        if isinstance(candidate, str) and _same_model(candidate, model_name):
            size = model.get("size", 0)
            size_vram = model.get("size_vram", 0)
            return ModelInMemory(
                name=candidate,
                size_bytes=int(size) if isinstance(size, int | float) else 0,
                size_vram_bytes=(
                    int(size_vram) if isinstance(size_vram, int | float) else 0
                ),
            )
    raise E2ETestFailure(
        f"Model {model_name!r} is absent from /api/ps after Crew kickoff. "
        f"Running models: {sorted(_model_names(running))}"
    )


def _add_no_proxy_host(base_url: str) -> None:
    """Ensure traffic to the local Ollama server bypasses HTTP proxies."""
    host = urlsplit(base_url).hostname
    if not host:
        return
    existing = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
    for required in ("localhost", "127.0.0.1", host):
        if required not in entries:
            entries.append(required)
    value = ",".join(entries)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _validate_embedding_evidence(calls: list[EmbeddingCallEvidence]) -> None:
    """Require both ingestion and query embedding calls with no CPU fallback."""
    _require(bool(calls), "No Sentence Transformer embedding calls were observed")
    phases = {call.phase for call in calls}
    _require("ingestion" in phases, "No knowledge-ingestion embedding call was observed")
    _require("query" in phases, "No knowledge-query embedding call was observed")

    for index, call in enumerate(calls, start=1):
        _require(
            bool(call.parameter_devices)
            and all(device.startswith("xpu") for device in call.parameter_devices),
            f"Embedding call {index} has non-XPU model parameters: {call.parameter_devices}",
        )
        _require(
            all(device.startswith("xpu") for device in call.buffer_devices),
            f"Embedding call {index} has non-XPU model buffers: {call.buffer_devices}",
        )
        _require(
            bool(call.forward_tensor_devices)
            and all(
                device.startswith("xpu") for device in call.forward_tensor_devices
            ),
            f"Embedding call {index} observed non-XPU forward tensors: "
            f"{call.forward_tensor_devices}",
        )
        _require(
            call.xpu_memory_allocated_bytes > 0,
            f"Embedding call {index} reported no allocated XPU memory",
        )


def _run_crew_rag(
    base_url: str,
    ollama_model: str,
    embedding_model: str,
    timeout: float,
    storage_dir: Path,
) -> tuple[str, dict[str, Any], str, str, list[EmbeddingCallEvidence]]:
    """Create knowledge, run Crew kickoff, and collect XPU evidence."""
    previous_storage_dir = os.getenv("CREWAI_STORAGE_DIR")
    os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    _add_no_proxy_host(base_url)

    try:
        import torch
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
            SentenceTransformerEmbeddingFunction,
        )
        from crewai import LLM, Agent, Crew, Task, TaskOutput
        from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
        from crewai.knowledge.knowledge_config import KnowledgeConfig
        from crewai.knowledge.source.string_knowledge_source import (
            StringKnowledgeSource,
        )
    except ImportError as error:
        raise E2ETestFailure(
            f"RAG dependencies could not be imported: {type(error).__name__}: {error}"
        ) from error

    _require(hasattr(torch, "xpu"), "This PyTorch build has no torch.xpu support")
    _require(torch.xpu.is_available(), "torch.xpu.is_available() returned False")

    # Suppress interactive first-run tracing prompts
    set_suppress_tracing_messages(True)

    calls: list[EmbeddingCallEvidence] = []
    original_call = SentenceTransformerEmbeddingFunction.__call__

    def observed_embedding_call(self: Any, input: Any) -> Any:
        model = getattr(self, "_model", None)
        _require(model is not None, "Sentence Transformer model is unavailable")
        parameter_devices = tuple(
            sorted({str(parameter.device) for parameter in model.parameters()})
        )
        buffer_devices = tuple(
            sorted({str(buffer.device) for buffer in model.buffers()})
        )
        forward_devices: set[str] = set()

        def observe_forward(_module: Any, args: tuple[Any, ...], output: Any) -> None:
            del _module
            forward_devices.update(_tensor_devices(args, torch))
            forward_devices.update(_tensor_devices(output, torch))

        hook = model.register_forward_hook(observe_forward)
        try:
            result = original_call(self, input)
            torch.xpu.synchronize()
        finally:
            hook.remove()

        texts = [str(value) for value in input]
        phase = "ingestion" if any(KNOWLEDGE_MARKER in text for text in texts) else "query"
        calls.append(
            EmbeddingCallEvidence(
                phase=phase,
                input_count=len(texts),
                parameter_devices=parameter_devices,
                buffer_devices=buffer_devices,
                forward_tensor_devices=tuple(sorted(forward_devices)),
                xpu_memory_allocated_bytes=torch.xpu.memory_allocated(),
            )
        )
        return result

    llm = LLM(
        model=f"ollama/{ollama_model}",
        base_url=base_url,
        temperature=0,
        max_tokens=180,
        timeout=timeout,
    )
    source = StringKnowledgeSource(content=PRIVATE_POLICY)
    embedder = {
        "provider": "sentence-transformer",
        "config": {
            "model_name": embedding_model,
            "device": "xpu",
            "normalize_embeddings": True,
        },
    }
    agent = Agent(
        role="Private Policy Analyst",
        goal="Answer policy questions only from retrieved private knowledge",
        backstory=(
            "You retrieve confidential policy records locally and copy exact codes, "
            "durations, and names without guessing."
        ),
        llm=llm,
        tools=[],
        knowledge_sources=[source],
        embedder=embedder,
        knowledge_config=KnowledgeConfig(results_limit=3, score_threshold=0.0),
        allow_delegation=False,
        max_iter=2,
        verbose=False,
    )

    # CrewAI validates this callback signature.
    # Keep the return type unannotated here to avoid false validation errors.
    def require_all_policy_fields(result: TaskOutput):  # type: ignore[no-untyped-def]
        """Retry incomplete answers without disclosing any expected fact value."""
        field_names = (
            (REQUIRED_FACTS[0], "authorization code"),
            (REQUIRED_FACTS[1], "credential lifetime"),
            (REQUIRED_FACTS[2], "final approver"),
        )
        missing_fields = [
            field_name for fact, field_name in field_names if fact not in result.raw
        ]
        if missing_fields:
            return (
                False,
                "The answer omitted these required fields: "
                f"{', '.join(missing_fields)}. Re-read the retrieved Additional "
                "Information and copy each exact value. Do not guess.",
            )
        return (True, result.raw)

    task = Task(
        description=(
            "Using only the retrieved private policy, state its exact authorization "
            "code, the lifetime of temporary contractor credentials, and the person "
            "who gives final approval. Include all three fields and copy each value "
            "exactly from Additional Information. Return one concise sentence."
        ),
        expected_output=(
            "One sentence containing the exact authorization code, credential "
            "lifetime, and final approver retrieved from knowledge."
        ),
        agent=agent,
        guardrail=require_all_policy_fields,
        guardrail_max_retries=2,
    )

    try:
        with patch.object(
            SentenceTransformerEmbeddingFunction,
            "__call__",
            observed_embedding_call,
        ):
            crew = Crew(
                agents=[agent],
                tasks=[task],
                memory=False,
                cache=False,
                verbose=False,
            )
            result = crew.kickoff()
    finally:
        if previous_storage_dir is None:
            os.environ.pop("CREWAI_STORAGE_DIR", None)
        else:
            os.environ["CREWAI_STORAGE_DIR"] = previous_storage_dir

    _require(not result.has_tool_failures, f"Crew recorded tool failures: {result.tool_failures}")
    query = agent.knowledge_search_query or ""
    context = agent.agent_knowledge_context or ""
    return result.raw, result.usage_metrics, query, context, calls


def _best_effort_shutdown() -> None:
    """Release global CrewAI state so script exits without waiting on background work."""
    try:
        from crewai.rag.config.utils import clear_rag_config

        clear_rag_config()
    except Exception:
        pass

    try:
        from crewai.events.event_bus import crewai_event_bus

        crewai_event_bus.shutdown(wait=False)
    except Exception:
        pass


def run_e2e(
    base_url: str,
    ollama_model: str,
    embedding_model: str,
    timeout: float,
) -> dict[str, Any]:
    """Run Ollama initial checks, combined RAG workflow, and all evidence checks."""
    api_root = _api_root(base_url)
    with httpx.Client(base_url=api_root, timeout=timeout, trust_env=False) as client:
        _ollama_initial_checks(client, ollama_model)
        temp_parent = os.getenv("TMPDIR") or "/tmp"
        with TemporaryDirectory(prefix="crewai-xpu-rag-", dir=temp_parent) as temp_dir:
            start = time.perf_counter()
            output, usage, query, context, embedding_calls = _run_crew_rag(
                base_url,
                ollama_model,
                embedding_model,
                timeout,
                Path(temp_dir),
            )
            elapsed = time.perf_counter() - start

        _require(bool(query), "CrewAI did not generate a knowledge search query")
        _require(bool(context), "CrewAI did not retrieve any agent knowledge context")
        context_missing = [fact for fact in REQUIRED_FACTS if fact not in context]
        _require(
            not context_missing,
            f"Retrieved knowledge context omitted facts {context_missing}: {context!r}",
        )
        output_missing = [fact for fact in REQUIRED_FACTS if fact not in output]
        _require(
            not output_missing,
            f"Final Crew output omitted facts {output_missing}: {output!r}",
        )
        _validate_embedding_evidence(embedding_calls)

        in_memory = _model_in_memory(client, ollama_model)
        _require(
            in_memory.size_vram_bytes > 0,
            "Ollama reports zero VRAM in memory; reject this as an entirely CPU run",
        )

    in_memory_evidence = asdict(in_memory)
    in_memory_evidence["vram_percent"] = in_memory.vram_percent
    return {
        "status": "PASS",
        "ollama_host": api_root,
        "ollama_model": ollama_model,
        "embedding_model": embedding_model,
        "embedding_device": "xpu",
        "required_facts": list(REQUIRED_FACTS),
        "knowledge_search_query": query,
        "retrieved_context": context,
        "crew_output": output,
        "crew_usage_metrics": usage,
        "elapsed_seconds": elapsed,
        "embedding_calls": [asdict(call) for call in embedding_calls],
        "ollama_model_in_memory": in_memory_evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Existing Ollama service URL.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        help="Installed Ollama model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"),
        help="Sentence Transformer model name or local model path.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="HTTP and LLM request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = run_e2e(
            args.host,
            args.ollama_model,
            args.embedding_model,
            args.timeout,
        )
    except E2ETestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"FAIL: RAG workflow raised {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        _best_effort_shutdown()

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
