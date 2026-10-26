# InferenceBench

InferenceBench measures how well a CLI coding agent can optimize an LLM inference server. Each agent gets a fixed time budget to write a server launcher for a base model. The harness then runs the launcher in a fresh container and measures serving speed under one of four scenarios. It also checks answer quality against a PyTorch reference. A Claude judge reviews the launcher and server log for disallowed shortcuts.

## Requirements

- Linux with Bash, Python 3.10 or newer, and Apptainer.
- An NVIDIA H100 80 GB, or an A100 with the `_a100` images.
- Network access and Apptainer fakeroot support for container builds.
- Shared storage for containers, model weights, dataset caches, and run outputs.
- Credentials for the selected agent and for the Claude judge.

The submission scripts use HTCondor and `condor_submit_bid`. To use another scheduler, adapt the `.sub` files and the submission commands.

## Configure storage

Run all commands from the repository root. Replace these paths with your storage directories:

```bash
export INFERENCE_BENCH_CONTAINERS_DIR="/path/to/shared/containers"
export INFERENCE_BENCH_RESULTS_DIR="/path/to/shared/results"
export HF_HOME="/path/to/shared/huggingface"
source src/commit_utils/set_env_vars.sh
mkdir -p "$INFERENCE_BENCH_CONTAINERS_DIR" "$INFERENCE_BENCH_RESULTS_DIR" "$HF_HOME"
```

Each job copies the selected model from `HF_HOME` into a private cache. Reserve disk space for one model copy per job. For gated models, set `HF_TOKEN`.

## Build containers

Build the agent container, the PyTorch reference container, the serving-engine containers you need, and the download container:

```bash
bash containers/build_container.sh inference
bash containers/build_backend_containers.sh torch vllm sglang
bash containers/build_container.sh download_hf_cache
```

Python package pins are in `containers/locks/`. For A100 nodes, build `inference_a100` and pass `--a100` to `build_backend_containers.sh`. The TGI image is the same for both GPU types.

## Download models and datasets

The resource file selects Mistral-7B-Instruct-v0.3, MMLU-Pro, and LongBench-v2. Edit it to use another model:

```bash
bash containers/download_hf_cache/download_hf_cache.sh \
  --resources-file containers/download_hf_cache/resources.json \
  --workers 4
```

Cache the benchmark samples for the default seeds:

```bash
apptainer exec \
  --no-home \
  --bind "$PWD:$PWD" \
  --bind "$HF_HOME:$HF_HOME" \
  --pwd "$PWD" \
  "$INFERENCE_BENCH_CONTAINERS_DIR/download_hf_cache.sif" \
  bash src/commit_utils/precache_seeds.sh
```

## Prepare the quality reference

The quality gate compares each agent's server with a PyTorch reference. This command submits one GPU job that computes the reference for quality seed `248` and 500 MMLU-Pro questions:

```bash
bash src/commit_utils/baselines/submit_precompute_all_baselines.sh \
  "mistralai/Mistral-7B-Instruct-v0.3" 40 torch 248 248 500 4
```

The reference is written to `src/eval/inference/baselines/quality/`. `commit.sh` refuses to submit runs until it exists. If you change the model, quality seed, or question count, compute a matching reference. For A100 nodes, add `--a100`.

To precompute baselines for another serving backend, replace `torch` with `vllm` or `sglang`. TGI cannot ignore end-of-sequence tokens, so it supports only quality references. Set `INFERENCE_BENCH_SKIP_SPEED=1` for a TGI submission.

## Configure credentials

Set the API key for the selected agent:

| Agent | Credential |
|---|---|
| `codex` | `OPENAI_API_KEY` |
| `claude` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `opencode` | Provider key in `agents/opencode/api_key` |

Git ignores credential files. Set them to mode `600`. OpenCode is installed only in the default H100 agent image.

The judge uses Claude Sonnet 4.6. Set `ANTHROPIC_API_KEY` or a Claude subscription token for every run, even when the agent uses another provider.

### Codex subscription

Enable device-code login in your ChatGPT security settings, then authenticate:

```bash
codex login --device-auth
cp ~/.codex/auth.json agents/codex_non_api/auth.json
chmod 600 agents/codex_non_api/auth.json
```

Select this adapter with `--agents codex_non_api:gpt-5-codex`.

### Claude subscription

Generate a token with `claude setup-token`, then save it without writing it to your shell history:

```bash
read -rs CLAUDE_CODE_OAUTH_TOKEN
printf '%s\n' "$CLAUDE_CODE_OAUTH_TOKEN" > agents/claude_non_api/oauth_token
chmod 600 agents/claude_non_api/oauth_token
unset CLAUDE_CODE_OAUTH_TOKEN
```

Select this adapter with `--agents claude_non_api:claude-sonnet-4-6`. The judge also uses this token file.

## Submit a run

Preview the jobs first:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-judge-key"
bash src/commit_utils/commit.sh \
  --agents codex:gpt-5.3-codex-high \
  --scenarios all \
  --num-hours 2 \
  --seed-pairs 21:1337 \
  --bid 40 \
  --dry-run
```

Remove `--dry-run` to submit one job per scenario. Each job gives the agent two hours, then runs the final evaluation. Add `--a100` to submit to A100 nodes.

The four scenarios are:

- `inference_scenario_a_input_heavy`
- `inference_scenario_b_output_heavy`
- `inference_scenario_c_high_load`
- `inference_scenario_d_general`

Each seed pair is `development_seed:evaluation_seed`. The agent tunes on requests from the development seed, and the final score uses the evaluation seed. Without `--seed-pairs`, the script submits `21:1337`, `248:428`, and `999:777`.

For fixed-output requests, the server must report `usage.completion_tokens` equal to the requested length. A request with a missing or different count fails.

Results are written to `INFERENCE_BENCH_RESULTS_DIR`.

## Matched-budget baseline

The matched-budget baseline searches vLLM and SGLang settings automatically with the same time budget as an agent. It uses the same default seed pairs as `commit.sh`. This command submits random search over vLLM for scenario A with a two-hour budget:

```bash
bash src/commit_utils/baselines/submit_hpo_search_baselines.sh \
  "mistralai/Mistral-7B-Instruct-v0.3" 40 random vllm a 7200
```

The arguments are the model, bid, search methods (`random`, `tpe`, `smac`), engines, scenarios (`a`–`d`), and budget in seconds. Each list is comma-separated. Use `--seed-pairs` to select other seed pairs.

## Harbor

To run InferenceBench on Modal cloud GPUs through Harbor instead of HTCondor, see [src/harbor_adapter/README.md](src/harbor_adapter/README.md).
