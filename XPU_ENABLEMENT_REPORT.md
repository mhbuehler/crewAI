# CrewAI XPU Enablement Report

## Summary

CrewAI is primarily an orchestration framework, not a model runtime. XPU enablement
will initially focus on three practical steps:

1. Validate Sentence Transformer embeddings on Intel XPU.
2. Run a complete CrewAI workflow against a local LLM server on Intel XPU.
3. Combine the validated embedding and LLM paths in a knowledge/RAG E2E test.

The first contribution, [PR-1](https://github.com/crewAIInc/crewAI/pull/6808),
documents `xpu` for Instructor and Sentence Transformer embeddings and tests that the
configuration is preserved. Real XPU validation is currently manual and all three
XPU script checks are passing; automated CI coverage for those checks is not in place yet.

## Repos Analyzed

| Repository | Role | Recommendation |
|---|---|---|
| [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | Core providers, RAG, knowledge, memory, and tests | Add configuration regression tests and focused hardware validation |
| [`crewAIInc/crewAI-examples`](https://github.com/crewAIInc/crewAI-examples) | Historical full examples | Archived and read-only since 2026-04-20 |
| [`crewAIInc/crewAI-quickstarts`](https://github.com/crewAIInc/crewAI-quickstarts) | Active notebook-based feature demos | Possible target for a concise XPU quickstart after the first E2E test is reproducible |
| [`crewAIInc/awesome-crewai`](https://github.com/crewAIInc/awesome-crewai) | Curated community links | Accepts only completed public projects |

## Testing

### Unit Test Analysis (Summary)

As of WW29, the separately tracked CrewAI unit test status:

| Metric | Count | Notes |
|---|---:|---|
| Total unit tests across run scope | 5,427 | Blocked modules expanded to individual tests, but parametrized tests in module-level errors could result in a larger actual total |
| Pure software unit tests | 5,427 | No GPU/XPU/CUDA-specific unit test paths, markers, or hardware dependencies |
| Passing pure software unit tests | 5,229 | Plus 1 xfail (not included in passing count) |
| Blocked pure software unit tests | 191 | 69 skipped + 122 collection/setup blocked tests |
| Failed pure software unit tests | 6 | Not related to XPU path coverage |
| XPU-runnable unit tests | 0 | No unit tests in this scope directly execute XPU-specific paths |
| Passing XPU-runnable unit tests | 0 | N/A |
| Blocked XPU-runnable unit tests | 0 | N/A |
| Failed XPU-runnable unit tests | 0 | N/A |
| Newly added XPU-specific unit tests | 0 | N/A |

**Key takeaway:** current CrewAI unit-test coverage in this run scope is
software-only. Unit tests provide useful regression signal, but do not prove XPU execution.

### Two Test Command Scopes

Two commands were used across sessions:

| Command | Intended scope | Observed role in this report |
|---|---|---|
| `uv run pytest lib/crewai/tests/ -x -q` | Targeted core framework tests | Primary command for PR validation and contributor guidance |
| `uv run pytest .` | Whole-repo discovery from root (includes CLI, tools) | Broader and potentially noisier for our purposes |

This report treats
`uv run pytest lib/crewai/tests/ -x -q` as the primary baseline command for
CrewAI framework validation, and references `uv run pytest .` as a broader,
optional repository-wide run which include the CLI, tools, files, and shared plumbing in the `crewai-core` package.

### Baseline Snapshot

The PR-1 validation snapshot from the targeted command:

| Check | Result | Meaning |
|---|---|---|
| `uv run pytest lib/crewai/tests/ -x -q` | **4,588 passed** | Regression baseline captured during PR-1 validation in WW32; not an XPU hardware test |

### XPU coverage status

| Test | Status |
|---|---|
| Sentence Transformer configuration compatibility | [PR-1](https://github.com/crewAIInc/crewAI/pull/6808) parameterizes the existing `cuda` test across all documented devices, adding `cpu`, `mps`, and `xpu` cases |
| Real Sentence Transformer embedding through CrewAI on XPU | Implemented; manual PASS |
| Full `Crew.kickoff()` against an XPU-backed local LLM | Implemented; manual PASS  |
| Knowledge/RAG with XPU embeddings and local LLM | Implemented; manual PASS |

An XPU test must confirm that the relevant model computation uses the Intel GPU. A
successful HTTP response alone is not sufficient evidence. Currently, these XPU checks are run manually via scripts and saved JSON evidence; they are not part of an automated CI.


## Contributions

| # | Contribution | Status | Acceptance criteria |
|---|---|---|---|
| PR-1 | [#6808: add XPU to embedding device options](https://github.com/crewAIInc/crewAI/pull/6808) | Open; checks passed and review feedback addressed | Maintainer approval and merge |
| PR-2 | Add embedding factory forwarding tests for `device="xpu"` | Proposed | Unit tests prove the downstream callable receives `xpu`; describe this as configuration coverage, not hardware support |
| Smoke Test | Validate Sentence Transformer embeddings on real XPU hardware | In Progress (manual PASS) | Valid vectors, verified model/device placement, and no CPU fallback |
| E2E-1 | Validate complete Crew kickoff against an XPU-backed local server | In Progress (manual PASS) | Correct output, confirmed Intel GPU use |
| E2E-2 | Validate knowledge/RAG with XPU embeddings and a local LLM | In Progress (manual PASS) | Correct retrieval and answers, confirmed Intel GPU use |

## XPU Test Plan

[Full README](scripts/README.md)

### Smoke Test: Sentence Transformer embeddings on XPU

**Goal:** Prove the local embedding path works on Intel XPU before building
E2E tests.

- Create an XPU-capable PyTorch environment and verify `torch.xpu.is_available()`
- Build a Sentence Transformer embedder through CrewAI with `device="xpu"`
- Generate valid embeddings for sample text
- Confirm XPU execution and reject CPU fallback

**Status:** [Script implemented](scripts/xpu_sentence_transformer_smoke.py) and passing manually ([output](scripts/xpu_sentence_transformer_smoke.json)).

### E2E-1: Private local summarization

**Use case:** summarize confidential support tickets, meeting notes, or incident
reports without sending their contents to a hosted model.

- Run a small instruct model with Ollama and verify it uses the Intel XPU
- Connect through CrewAI's Ollama provider
- Use one agent, one task, and simple input with no API keys or web tools
- Run `Crew.kickoff()` and verify required facts appear in the result
- Confirm Intel GPU execution and reject an entirely CPU inference run

**Status:** [Script implemented](scripts/xpu_ollama_crew_e2e.py) and passing manually ([output](scripts/xpu_ollama_crew_e2e.json)).

### E2E-2: Private policy-document Q&A

**Use case:** answer employee questions from internal policies or product manuals
while keeping documents and inference local.

- Prepare a small test document containing unique facts
- Ingest it through CrewAI knowledge using the validated Sentence Transformer XPU
  configuration
- Ask questions whose answers exist only in the test document
- Verify the answer contains facts available only in the test document
- Confirm XPU embedding execution and Intel GPU acceleration for Ollama

**Status:** [Script implemented](scripts/xpu_rag_ollama_e2e.py) and passing manually ([output](scripts/xpu_rag_ollama_e2e.json)).

### Upstreaming

- Submit planned hardware-independent configuration tests to the main CrewAI repository (PR-1 & PR-2)
- Propose a concise `crewAI-quickstarts` notebook after E2E tests are stable
- Use a standalone repository if CrewAI has no suitable home for hardware-dependent
  tests

## Next Steps

- [x] Run the CrewAI unit suite: 4,588 tests passed
- [x] Analyze CrewAI's repo for vendor-specific text, locating local embedding and LLM provider paths
- [x] Analyze the examples, quickstarts, and community repositories
- [x] Prepare and submit PR documenting XPU embedding device options
- [ ] Follow up on PR-1 until merged
- [x] Create an isolated XPU environment, run a Sentence Transformer embedding smoke test through CrewAI, and confirm XPU utilization
- [ ] Prepare and submit PR-2 testing embedding factory forwarding
- [x] Create a repeatable setup for running the XPU E2E tests
- [x] Implement E2E-1: private local summarization
- [ ] Ask `crewAI-quickstarts` maintainers whether a companion XPU notebook fits their
      preferred scope and directory structure
- [ ] Submit E2E test(s) as notebook(s) to the quickstarts repo if maintainers agree
- [x] Implement E2E-2: private policy-document Q&A with XPU embeddings

## Future Work

- **ONNX/OpenVINO:** Validate `OpenVINOExecutionProvider` on Intel GPU, verify CrewAI
  forwards the provider configuration, and add tests and documentation.
- **OpenCLIP:** Validate text and image embeddings with `device="xpu"`, confirm model
  and tensor placement, and add configuration tests and documentation.
