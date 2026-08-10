# ruff: noqa: T201
"""Validate a complete CrewAI kickoff against existing XPU Ollama service.

This manual E2E test expects the Ollama service and model to be already running,
executes one agent and one task with no tools, verifies facts in the final Crew
output, and rejects an entirely CPU-backed run by requiring Ollama's /api/ps
endpoint to report nonzero VRAM for the model while it is running.

Example:
    export OLLAMA_HOST="http://localhost:11434"
    export OLLAMA_MODEL="qwen2.5:0.5b"
    python scripts/xpu_ollama_crew_e2e.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


PRIVATE_INCIDENT_NOTE = (
    "Private incident note: The incident identifier is BLUE-COMET. "
    "The only affected service was relay-cache. "
    "Recovery completed at 09:17 UTC. "
    "Do not infer facts that are not written here."
)
REQUIRED_FACTS = ("BLUE-COMET", "relay-cache", "09:17 UTC")


class E2ETestFailure(RuntimeError):
    """Raised when the Ollama-backed CrewAI E2E test fails."""


@dataclass(frozen=True)
class ModelInMemory:
    """Relevant model in-memory evidence returned by Ollama."""

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
    """Compare Ollama model names while tolerating an implicit latest tag."""
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


def _initial_checks(client: httpx.Client, model_name: str) -> None:
    tags = _get_json(client, "/api/tags")
    installed = _model_names(tags)
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
    running_names = sorted(_model_names(running))
    raise E2ETestFailure(
        f"Model {model_name!r} is absent from /api/ps after Crew kickoff. "
        f"Running models: {running_names}"
    )


def _add_no_proxy_host(base_url: str) -> None:
    """Ensure traffic to Ollama bypasses configured HTTP proxies."""
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


def _run_crew(base_url: str, model_name: str, timeout: float) -> tuple[str, dict[str, Any]]:
    """Execute one private summarization task through CrewAI and Ollama."""
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    _add_no_proxy_host(base_url)

    try:
        from crewai import LLM, Agent, Crew, Task
    except ImportError as error:
        raise E2ETestFailure(
            f"CrewAI could not be imported: {type(error).__name__}: {error}"
        ) from error

    llm = LLM(
        model=f"ollama/{model_name}",
        base_url=base_url,
        temperature=0,
        max_tokens=160,
        timeout=timeout,
    )
    agent = Agent(
        role="Private Incident Summarizer",
        goal="Summarize the supplied incident note without adding or omitting facts",
        backstory=(
            "You process confidential operational notes locally and copy identifiers, "
            "service names, and timestamps exactly."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        max_iter=2,
        verbose=False,
    )
    task = Task(
        description=(
            "Read this confidential note:\n\n{incident_note}\n\n"
            "Return one concise sentence containing the exact incident identifier, "
            "affected service, and recovery time. Do not add other facts."
        ),
        expected_output=(
            "One sentence containing incident id, service, and recovery time."
        ),
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        memory=False,
        cache=False,
        verbose=False,
    )

    result = crew.kickoff(inputs={"incident_note": PRIVATE_INCIDENT_NOTE})
    _require(not result.has_tool_failures, f"Crew recorded tool failures: {result.tool_failures}")
    return result.raw, result.usage_metrics


def run_e2e(base_url: str, model_name: str, timeout: float) -> dict[str, Any]:
    """Run initial checks, Crew kickoff, result checks, and GPU memory checks."""
    api_root = _api_root(base_url)
    with httpx.Client(base_url=api_root, timeout=timeout, trust_env=False) as client:
        _initial_checks(client, model_name)
        start = time.perf_counter()
        output, usage = _run_crew(base_url, model_name, timeout)
        elapsed = time.perf_counter() - start

        missing = [fact for fact in REQUIRED_FACTS if fact not in output]
        _require(not missing, f"Crew output omitted required facts {missing}: {output!r}")

        in_memory = _model_in_memory(client, model_name)
        _require(
            in_memory.size_vram_bytes > 0,
            "Ollama reports zero VRAM in memory; reject this as an entirely CPU run",
        )

    in_memory_evidence = asdict(in_memory)
    in_memory_evidence["vram_percent"] = in_memory.vram_percent
    return {
        "status": "PASS",
        "ollama_host": api_root,
        "ollama_model": model_name,
        "required_facts": list(REQUIRED_FACTS),
        "crew_output": output,
        "crew_usage_metrics": usage,
        "elapsed_seconds": elapsed,
        "ollama_model_in_memory": in_memory_evidence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Existing Ollama service URL (default: OLLAMA_HOST or localhost:11434).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        help="Installed Ollama model (default: OLLAMA_MODEL or qwen2.5:0.5b).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP and LLM request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        evidence = run_e2e(args.host, args.model, args.timeout)
    except E2ETestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"FAIL: Crew kickoff raised {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())