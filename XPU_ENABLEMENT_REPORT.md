# CrewAI XPU Enablement Report

## Summary

CrewAI is an orchestration framework, not a model runtime. XPU enablement
focuses on four things:

1. Validate Sentence Transformer embeddings on Intel XPU.
2. Validate OpenCLIP text and image embeddings on Intel XPU.
3. Run a complete CrewAI workflow against a local LLM server on Intel XPU.
4. Combine the validated embedding and LLM paths in a knowledge/RAG E2E test.

The first proposal, [Issue #7173](https://github.com/crewAIInc/crewAI/issues/7173),
was completed through
[PR #6808: docs(rag): add xpu to embedding device options](https://github.com/crewAIInc/crewAI/pull/6808).
It documents `xpu` for Instructor and Sentence Transformer embeddings and tests
that the configuration is preserved. The follow-up proposal,
[Issue #7921](https://github.com/crewAIInc/crewAI/issues/7921), was accepted by a
maintainer, and
[PR #7923: docs(rag): add OpenCLIP provider device options](https://github.com/crewAIInc/crewAI/pull/7923)
is open to add the equivalent OpenCLIP documentation, provider metadata, and
configuration coverage. Real XPU validation is currently manual, and all four
validation scripts are passing. The automated E2E RAG CI
[integration plan](#ci-integration-plan-knowledgerag-e2e) is documented below
and tracked separately.

## Repos Analyzed

| Repository | Role | Recommendation |
|---|---|---|
| [`crewAIInc/crewAI`](https://github.com/crewAIInc/crewAI) | Core providers, RAG, knowledge, memory, and tests | Add configuration regression tests and focused hardware validation |
| [`crewAIInc/crewAI-examples`](https://github.com/crewAIInc/crewAI-examples) | Historical full examples | Archived and read-only since 2026-04-20 |
| [`crewAIInc/crewAI-quickstarts`](https://github.com/crewAIInc/crewAI-quickstarts) | Active notebook-based feature demos | [#7: private local RAG quickstart with Ollama](https://github.com/crewAIInc/crewAI-quickstarts/pull/7) submitted |
| [`crewAIInc/awesome-crewai`](https://github.com/crewAIInc/awesome-crewai) | Curated community links | Accepts only completed public projects |

## Testing

### Unit Test Scope

The primary reporting scope is `lib/crewai/tests/` because it is the scope
called "all tests" in the contributor guide, it is the core framework test
directory exercised by the pull-request workflow, and it contains the test
changed by [PR #6808](https://github.com/crewAIInc/crewAI/pull/6808). Counts from
broader scopes must be reported separately.

| Command | Scope | Use |
|---|---|---|
| `uv run pytest lib/crewai/tests/ -x -q` | Core CrewAI framework package | Exact contributor-guide smoke command; stops at the first failure, so do not use it to calculate outcome totals |
| `uv run pytest lib/crewai/tests/` | Core CrewAI framework package | Primary scope for the fresh unit-test analysis |
| `uv run pytest` | Five package test roots configured in `pyproject.toml` | Optional workspace-wide follow-up covering CrewAI, tools, files, CLI, and core |
| `uv run pytest .` | Unbounded recursive discovery, including paths such as `lib/devtools/tests` that are not in configured `testpaths` | Do not use for the headline unit-test metrics |

### Fresh Manual Run

Run from a clean checkout of the desired upstream `main` revision. Use the
project environment rather than an unrelated active virtual environment, and
record the revision and tool versions with the result.

```bash
git rev-parse HEAD
git status --short
uname -a
nproc
uv --version
uv sync --python 3.12 --all-groups --all-extras --all-packages
uv run --python 3.12 python --version

mkdir -p unit-test-artifacts
set -o pipefail
uv run --python 3.12 pytest lib/crewai/tests/ \
  -ra \
  --tb=long \
  --junitxml=unit-test-artifacts/crewai-py312.xml \
  2>&1 | tee unit-test-artifacts/crewai-py312.log
```

This deliberately omits `-x` so every outcome is reported. `-ra` prints skip,
xfail, failure, and error reasons; `--tb=long` preserves import and fixture
tracebacks. Do not set real provider credentials or start local services for
the baseline, because doing so changes which integration tests skip.

If the run reports collection or setup errors, rerun only the affected files
serially with `-n 0 -vv -ra --tb=long`. Treat that as diagnostic evidence, not
as a replacement baseline.

### Results Tracking

The internal [tracker](https://intel.sharepoint.com/:x:/r/sites/appliedaiframeworks/Shared%20Documents/Post%20Training%20FWKs/XPU%20enablement/XPU%20enablement%20tracking.xlsx?d=w31b4d0b908594666a0e28781d22de320&csf=1&web=1&e=8sTqPD) is the source of truth for UT status and results by work week. For a blocked test breakdown, see the [wiki page](https://wiki.ith.intel.com/spaces/posttraining/pages/4973765370/Agentic+Fwks+UT+Status).

### XPU Coverage Status

| Test | Status |
|---|---|
| Sentence Transformer configuration compatibility | [PR #6808](https://github.com/crewAIInc/crewAI/pull/6808) parameterizes the existing `cuda` test across all documented devices, adding `cpu`, `mps`, and `xpu` cases |
| OpenCLIP configuration compatibility | [PR #7923](https://github.com/crewAIInc/crewAI/pull/7923) is open to document `mps` and `xpu` and test that all documented device strings are preserved |
| Real Sentence Transformer embedding through CrewAI on XPU | Implemented; manual PASS |
| OpenCLIP text and image embeddings through CrewAI on XPU | Implemented; manual PASS |
| Full `Crew.kickoff()` against an XPU-backed local LLM | Implemented; manual PASS |
| Knowledge/RAG with XPU embeddings and local LLM | Implemented; manual PASS |

All XPU tests confirm that the relevant model computation uses the Intel GPU.
Currently, they are run manually via scripts with saved JSON evidence. They are
not part of automated CI, though an integration plan is documented below and
tracked [here](https://intel.sharepoint.com/:x:/r/sites/appliedaiframeworks/Shared%20Documents/Post%20Training%20FWKs/XPU%20enablement/XPU%20enablement%20tracking.xlsx?d=w31b4d0b908594666a0e28781d22de320&csf=1&web=1&e=yhLeGp).

### CI Integration Plan: Knowledge/RAG E2E

The first automated CrewAI case should be based on the flow validated by
[`xpu_rag_ollama_e2e.py`](scripts/xpu_rag_ollama_e2e.py). The
script uses Ollama's native `/api/tags` and `/api/ps` endpoints, including
`size_vram`, as part of its pass criteria, so if vLLM is chosen as the target local LLM service,
it will require some refactoring.

Recommended job flow:

1. Schedule the job on a dedicated Intel GPU runner and expose the required
  devices to the job and model service.
2. Start a pinned XPU-enabled Ollama build, restore the model cache, pull the pinned
  model when absent, and wait for `/api/tags` to become ready.
3. Install the pinned PyTorch XPU wheel before installing the local CrewAI packages
  with `--no-deps`; do not use the repository's CPU PyTorch for this job.
4. Run the E2E script with `OLLAMA_HOST`, `OLLAMA_MODEL`, and
  `SENTENCE_TRANSFORMER_MODEL` set. For a vLLM adaptation, supply the
  corresponding settings. Preserve the process exit code as the job result.
5. Publish the JSON evidence, console log, model server log, package versions, hardware
  inventory, and JUnit XML as artifacts even when the test fails.
6. Stop the service and clean temporary CrewAI storage while retaining model caches
  managed by the runner.


## Contributions

| Contribution | Issue | Status | Acceptance criteria |
|---|---|---|---|
| [PR #6808: docs(rag): add xpu to embedding device options](https://github.com/crewAIInc/crewAI/pull/6808) | [#7173: include all supported RAG embedding devices](https://github.com/crewAIInc/crewAI/issues/7173) (closed as completed) | Merged | Documentation and provider metadata list `cpu`, `cuda`, `mps`, and `xpu`; configuration tests preserve every documented value |
| [PR #7923: docs(rag): add OpenCLIP provider device options](https://github.com/crewAIInc/crewAI/pull/7923) | [#7921: include all supported OpenCLIP device options](https://github.com/crewAIInc/crewAI/issues/7921) (open; maintainer invited PR) | Open | Documentation and provider metadata list `cpu`, `cuda`, `mps`, and `xpu`; configuration tests preserve every documented value |
| Factory forwarding test proposal | N/A | Not pursued | The shared factory path has no device-specific logic; provider configuration tests and manual hardware validation cover the relevant risk |
| Sentence Transformer smoke test | N/A | Complete (manual PASS) | Valid vectors, verified model/device placement, and no CPU fallback |
| OpenCLIP smoke test | N/A | Complete (manual PASS) | Valid vectors, shared embedding space, verified model/tensor placement, and no CPU fallback |
| E2E-1 | N/A | Complete (manual PASS) | Correct output, confirmed Intel GPU use |
| E2E-2 | N/A | Complete (manual PASS) | Correct retrieval and answers, confirmed Intel GPU use |
| [crewAI-quickstarts PR #7: feat: add private local RAG quickstart with Ollama](https://github.com/crewAIInc/crewAI-quickstarts/pull/7) | N/A | Open | E2E-2 adapted into a reusable notebook and submitted upstream |

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

### Smoke Test: OpenCLIP text and image embeddings on XPU

**Goal:** Prove both OpenCLIP encoder paths run through CrewAI on Intel XPU.

- Build the OpenCLIP `ViT-B-32` embedder through CrewAI with `device="xpu"`
- Generate normalized text and image embeddings in the shared 512-dimensional space
- Confirm model parameters, buffers, and forward tensors are on `xpu:0`
- Confirm nonzero XPU memory allocation and reject CPU fallback

**Status:** [Script implemented](scripts/xpu_openclip_smoke.py) and passing manually ([output](scripts/xpu_openclip_smoke.json)).

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

- Filed the feature requests for the Sentence Transformer and Instructor work
  ([Issue #7173](https://github.com/crewAIInc/crewAI/issues/7173)) and the
  OpenCLIP follow-up ([Issue #7921](https://github.com/crewAIInc/crewAI/issues/7921))
- Submitted configuration and documentation contributions
  ([PR #6808](https://github.com/crewAIInc/crewAI/pull/6808) and
  [PR #7923](https://github.com/crewAIInc/crewAI/pull/7923))
- Submitted [#7: private local RAG quickstart with Ollama](https://github.com/crewAIInc/crewAI-quickstarts/pull/7)
  to `crewAI-quickstarts`
- Identified a standalone repository as the fallback if CrewAI has no suitable
  home for hardware-dependent tests

## Next Steps

- [x] Capture the unit-test baseline in the internal tracker
- [x] Rerun `lib/crewai/tests/` on the current `main` revision and update the unit test counts in the internal tracker
- [x] Analyze CrewAI's repo for vendor-specific text, locating local embedding and LLM provider paths
- [x] Analyze the examples, quickstarts, and community repositories
- [x] File [Issue #7173](https://github.com/crewAIInc/crewAI/issues/7173) and submit [PR #6808](https://github.com/crewAIInc/crewAI/pull/6808) documenting XPU embedding device options
- [x] Follow up on [PR #6808](https://github.com/crewAIInc/crewAI/pull/6808) until merged
- [x] Create an isolated XPU environment, run a Sentence Transformer embedding smoke test through CrewAI, and confirm XPU utilization
- [x] Implement E2E-1: private local summarization
- [x] Implement E2E-2: private policy-document Q&A with XPU embeddings
- [x] Validate OpenCLIP text and image embeddings with `device="xpu"`, confirm model and tensor placement
- [x] File [Issue #7921](https://github.com/crewAIInc/crewAI/issues/7921) and submit [PR #7923](https://github.com/crewAIInc/crewAI/pull/7923) with OpenCLIP configuration tests and documentation
- [x] Evaluate embedding factory forwarding coverage and decide no separate PR is needed
- [x] Create a repeatable setup for running the XPU E2E tests
- [x] Submit the E2E test as [crewAI-quickstarts PR #7](https://github.com/crewAIInc/crewAI-quickstarts/pull/7)
- [x] Add E2E-2 local RAG test to internal CI plan
- [ ] Follow up on [PR #7923](https://github.com/crewAIInc/crewAI/pull/7923) review feedback until merged
