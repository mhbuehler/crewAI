# CrewAI XPU Enablement Report

## Summary

CrewAI is primarily an orchestration framework, not a model runtime. XPU enablement
will initially focus on three practical steps:

1. Validate Sentence Transformer embeddings on Intel XPU.
2. Run a complete CrewAI workflow against a local LLM server on Intel XPU.
3. Combine the validated embedding and LLM paths in a knowledge/RAG E2E test.

The first contribution, [PR-1](https://github.com/crewAIInc/crewAI/pull/6808),
documents `xpu` for Instructor and Sentence Transformer embeddings and tests that the
configuration is preserved. Real XPU inference and no-CPU-fallback behavior still
need hardware validation.

## Repos Analyzed

| Repository | Role | Recommendation |
|---|---|---|
| [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | Core providers, RAG, knowledge, memory, and tests | Add configuration regression tests and focused hardware validation |
| [`crewAIInc/crewAI-examples`](https://github.com/crewAIInc/crewAI-examples) | Historical full examples | Archived and read-only since 2026-04-20 |
| [`crewAIInc/crewAI-quickstarts`](https://github.com/crewAIInc/crewAI-quickstarts) | Active notebook-based feature demos | Possible target for a concise XPU quickstart after the first E2E test is reproducible |
| [`crewAIInc/awesome-crewai`](https://github.com/crewAIInc/awesome-crewai) | Curated community links | Accepts only completed public projects |

## Testing

### Baseline

The following result was recorded during PR-1 validation.

| Check | Result | Meaning |
|---|---|---|
| `uv run pytest lib/crewai/tests/ -x -q` | **4,588 passed** | General regression baseline; not an XPU hardware test |

### XPU coverage status

| Test | Status |
|---|---|
| Sentence Transformer configuration compatibility | [PR-1](https://github.com/crewAIInc/crewAI/pull/6808) parameterizes the existing `cuda` test across all documented devices, adding `cpu`, `mps`, and `xpu` cases |
| Real Sentence Transformer embedding through CrewAI on XPU | Planned as Smoke Test |
| Full `Crew.kickoff()` against an XPU-backed local LLM | Planned as E2E-1 |
| Knowledge/RAG with XPU embeddings and local LLM | Planned as E2E-2 |

An XPU test must confirm that the relevant model computation uses the Intel GPU. A
successful HTTP response alone is not sufficient evidence.

## Contributions

| # | Contribution | Status | Acceptance criteria |
|---|---|---|---|
| PR-1 | [#6808: add XPU to embedding device options](https://github.com/crewAIInc/crewAI/pull/6808) | Open; checks passed and review feedback addressed | Maintainer approval and merge |
| PR-2 | Add embedding factory forwarding tests for `device="xpu"` | Planned | Unit tests prove the downstream callable receives `xpu`; describe this as configuration coverage, not hardware support |
| Smoke Test | Validate Sentence Transformer embeddings on real XPU hardware | Planned | Valid vectors, verified model/device placement, and no CPU fallback |
| E2E-1 | Validate complete Crew kickoff against an XPU-backed local server | Planned | Correct output, confirmed Intel GPU use |
| E2E-2 | Validate knowledge/RAG with XPU embeddings and a local LLM | Planned | Correct retrieval and answers, confirmed Intel GPU use |

## XPU Test Plan

### Smoke Test: Sentence Transformer embeddings on XPU

**Goal:** Prove the local embedding path works on Intel XPU before building either
E2E test.

- Create an XPU-capable PyTorch environment and verify `torch.xpu.is_available()`.
- Build a Sentence Transformer embedder through CrewAI with `device="xpu"`.
- Generate valid embeddings for sample text.
- Confirm XPU execution and reject CPU fallback.

### E2E-1: Private local summarization

**Use case:** summarize confidential support tickets, meeting notes, or incident
reports without sending their contents to a hosted model.

- Run a small instruct model with Ollama and verify it uses the Intel XPU.
- Connect through CrewAI's Ollama provider.
- Use one agent, one task, and simple input with no API keys or web tools.
- Run `Crew.kickoff()` and verify required facts appear in the result.
- Confirm Intel GPU execution and reject an entirely CPU inference run.

### E2E-2: Private policy-document Q&A

**Use case:** answer employee questions from internal policies or product manuals
while keeping documents and inference local.

- Prepare a small test document containing unique facts.
- Ingest it through CrewAI knowledge using the validated Sentence Transformer XPU
  configuration.
- Ask questions whose answers exist only in the test document.
- Verify the answer contains facts available only in the test document.
- Confirm XPU embedding execution and Intel GPU acceleration for Ollama.

Run this only after the Sentence Transformer hardware validation and E2E-1 are stable.

### Upstreaming

- Submit planned hardware-independent regression tests to the main CrewAI repository (PR-1 & PR-2).
- Propose a concise `crewAI-quickstarts` notebook after E2E-1 is reproducible.
- Use a standalone repository if CrewAI has no suitable home for hardware-dependent
  tests.

## Next Steps

- [x] Run the CrewAI unit suite: 4,588 tests passed.
- [x] Analyze CrewAI's repo for vendor-specific text, locating local embedding and LLM provider paths.
- [x] Analyze the examples, quickstarts, and community repositories.
- [x] Prepare and submit PR documenting XPU embedding device options.
- [ ] Follow up on PR-1 until merged.
- [ ] Create an isolated XPU environment, run a Sentence Transformer embedding smoke test through CrewAI, and confirm XPU utilization.
- [ ] Prepare and submit PR-2 testing embedding factory forwarding.
- [ ] Create a repeatable setup for running the XPU E2E tests.
- [ ] Implement E2E-1: private local summarization.
- [ ] Ask `crewAI-quickstarts` maintainers whether a companion XPU notebook fits their
      preferred scope and directory structure.
- [ ] Submit the quickstart if maintainers agree and E2E-1 is reproducible.
- [ ] Implement E2E-2: private policy-document Q&A with XPU embeddings.

## Future Work

- **ONNX/OpenVINO:** Validate `OpenVINOExecutionProvider` on Intel GPU, verify CrewAI
  forwards the provider configuration, and add tests and documentation.
- **OpenCLIP:** Validate text and image embeddings with `device="xpu"`, confirm model
  and tensor placement, and add configuration tests and documentation.
