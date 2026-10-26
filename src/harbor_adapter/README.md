# InferenceBench Harbor Adapter

This adapter runs InferenceBench on cloud GPUs through [Harbor](https://docs.harborframework.com) and [Modal](https://modal.com). The reference pipeline runs on HTCondor and Apptainer. This document calls that pipeline "condor". The adapter uses the same prompt, evaluator, quality gate, and disallowed-usage judge as condor. [Differences from condor](#differences-from-condor) lists the differences.

## What you need

- A Modal account. Each task uses one H100 GPU for the agent and one for the verifier.
- A read-only Hugging Face token with access to `mistralai/Mistral-7B-Instruct-v0.3`.
- `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` for the judge. To run claude-code on a Claude subscription, also set `CLAUDE_FORCE_OAUTH=1`; Harbor then keeps `ANTHROPIC_API_KEY` out of the agent sandbox.
- The key of your agent: `ANTHROPIC_API_KEY` (claude-code), `OPENAI_API_KEY` (codex), `GEMINI_API_KEY` (gemini-cli), or `OPENCODE_API_KEY` (opencode).
- The baselines under `src/eval/inference/baselines/`. Follow "Prepare the quality reference" in the repository README. Task generation stops when the quality reference, the speed baseline, or a sample file for a seed is missing.
- [uv](https://docs.astral.sh/uv/).

## Setup

1. Install Harbor and log in to Modal:

    ```bash
    uv tool install --force 'harbor[modal]>=0.25.0'
    harbor_py="$(dirname "$(readlink -f "$(command -v harbor)")")/python"
    $harbor_py -m modal setup
    ```

2. Copy `example.env` to `.env` at the repository root and fill in the keys. The scripts load only the keys that they need. A variable that you export wins over `.env`, except `HF_TOKEN`: there, `.env` wins. `INFERENCE_BENCH_ENV_FILE` selects another file.

## Run

One command generates the tasks and runs them:

```bash
cd src/harbor_adapter
bash run_modal_task.sh --scenario all --base-model all \
    --agent claude-code --model anthropic/claude-sonnet-4-6 --num-hours 2 --job-name sweep1
```

- `--scenario` takes `a`, `b`, `c`, `d`, a comma-separated list, or `all`. `--base-model` takes `mistral-7b` or `all`.
- `--seed-pairs` takes `agent_seed:eval_seed` pairs. The default is `DEFAULT_SEED_PAIRS` from `src/commit_utils/set_env_vars.sh` (`21:1337,248:428,999:777`). The agent seed sets `INFERENCE_BENCH_DATASET_SEED` in the agent sandbox. The evaluation seed sets it in the verifier.
- The tasks go to `tasks/<job-name>/`. Each task runs as its own Harbor job in `jobs/<job-name>/<short>/`, with its log in `jobs/<job-name>/<short>.log`. `<short>` is the task name without the `inferencebench-` prefix, for example `a-mistral-7b-21-1337`. `--parallel N` limits concurrent tasks. The command prints a table of exit codes and rewards at the end.
- Each task gets a Modal volume, `ib-<job-name>-<task>`. The volume keeps the agent's home directory after the run. `--delete-volume` removes it.
- Extra `harbor run` flags go after `--`, for example `-- --agent-timeout-multiplier 0.1`.

To generate tasks without running them, or to run one task in the foreground:

```bash
python3 run_adapter.py --scenario a --seed-pairs 21:1337 --output tasks/manual
bash run_modal_task.sh --task tasks/manual/inferencebench-a-mistral-7b-21-1337 \
    --agent claude-code --model anthropic/claude-sonnet-4-6 --job-name run1
```

Generate tasks for the agent that runs them (`run_adapter.py --agent`), because the prompt has an agent-specific clause.

### Agent options

| condor `agents/*/solve.sh` | `run_modal_task.sh` |
|---|---|
| CLI version from `containers/inference.def` | the same pin, read from `template/Dockerfile`; `--cli-version` or `CLAUDE_CLI_VERSION`, `CODEX_CLI_VERSION`, `GEMINI_CLI_VERSION`, `OPENCODE_CLI_VERSION` override it (`latest` is resolved once with `npm view`) |
| `BASH_MAX_TIMEOUT_MS=36000000` (`agents/common.sh`) | `--ae BASH_MAX_TIMEOUT_MS=36000000` for claude-code only |
| codex model suffix `-high`, `-med`, `-low` | stripped from `--model` and passed as `--ak reasoning_effort=...` |
| `codex --search` | `--ak web_search=live` |
| `opencode.json` in `agents/opencode/solve.sh` | extracted from that file and passed as `--ak opencode_config=...` |

`--effort` sets the reasoning effort for claude-code and codex. `--ak key=value` passes any other Harbor agent option.

## Results

Harbor writes `reward.json` in the verifier directory of each trial:

| Key | Value |
|---|---|
| `reward` | the scenario's primary metric from `primary_metric` in `src/eval/inference/hpo_search_baselines.py` (A: 1/TTFT p50, B: 1/TPOT p50, C: geometric mean of request throughput over the burst, poisson, and constant profiles, D: geometric mean of 1/TTFT p50, 1/TPOT p50, and request throughput of the burst profile). It is 0 when the evaluation is invalid, the speed check fails, or the quality gate fails. |
| `eval_valid`, `speed_pass`, `quality_pass` | 1 or 0 |
| `judge_valid` | 1 when the judge exited 0 and wrote valid verdict files; condor reports this through the job exit code |
| `contamination_detected`, `disallowed_use_detected` | 1 or 0 from the judge; absent when the judge wrote no valid verdict |

The judge verdicts do not change `reward`. Condor also keeps the verdicts separate from `metrics.json`.

To export the trials into the condor results layout under `INFERENCE_BENCH_RESULTS_DIR`:

```bash
python3 harbor_to_results.py jobs/sweep1
```

Each trial becomes `<results>/<agent>_<model>_<N>h/<scenario>_<model>_<run-id>/`. The run ID is the trial start time in Unix seconds plus the Harbor trial suffix, so parallel trials do not collide. The directory has the condor files (`metrics.json`, `eval.log`, `server.log`, `metrics_preview.json`, the judgement files, `final_runtime_env.sh`, `task/`, `prompt.txt`, `time_taken.txt`, `run_config.json`, `solve_out_run1.txt`, `output.log`) and a `harbor/` directory with `result.json`, `config.json`, and `collect.log` (the hand-off log, which Harbor stores under `artifacts/logs/artifacts/`).

`task/` is a capped copy of the agent's task directory: files of at most 64 MiB, at most 1 GiB and 20000 files in total, smallest first, without weight files, caches, virtual environments, and `.git`. `task/.snapshot_dropped.txt` lists what was left out. The cap keeps the verifier log download inside Harbor's 600 s archive limit.

## How it works

Each task has this layout:

```
inferencebench-a-mistral-7b-21-1337/
├── task.toml        resources, timeouts, per-task env, collect hook
├── instruction.md   prompt from src/eval/general/get_prompt.py
├── environment/     agent image: home/task/ (condor task files), inference_eval/ (evaluator bundle), collect.sh
└── tests/           verifier image: test.sh, ib/ (evaluator, scenario, judge, baselines)
```

Both Dockerfiles start with `template/Dockerfile`, which mirrors `containers/inference.def`. The verifier image is the same base plus the baked evaluator, baselines, and judge files. The agent's launcher therefore runs in the same software stack as the fresh evaluation container in condor.

1. **Agent setup.** The healthcheck downloads the base model into `/hf_cache` (a link to `/home/agent/hf_cache`) and then writes `/timer_start`. `timer.sh` counts down from that time. The download does not use the agent budget, as in condor, where the job copies a cached model before the agent starts.
2. **Agent phase.** The agent runs as the non-root user `agent` (`[agent] user`, `HOME=/home/agent`), as condor runs it as the job user. System site-packages are read-only, so `pip install` goes to `~/.local`, as in condor. The agent works in `/home/agent/task` with the condor file layout and the condor server and cache variables (`HOST`, `PORT`, `INFERENCE_BENCH_*`, `HF_HOME`, `PYTHONUSERBASE=/home/agent/.local`). Harbor sends the prompt directly, so `PROMPT_FILE`, `AGENT_CONFIG`, `INFERENCE_BENCH_RUN_INDEX`, and `INFERENCE_BENCH_REMAINING_SECONDS` are not set.
3. **Hand-off.** The `[[verifier.collect]]` hook runs `collect.sh` in the agent sandbox after the agent exits. It captures the environment of the live server into `final_runtime_env.sh` (the capture code comes from `src/run_task.sh`). It copies the `/tmp` targets of symbolic links in the task directory. It then copies `/home/agent` (task directory, user packages, other HF cache entries) to the Modal volume at `/mnt/ib_handoff`, which both sandboxes mount, and writes a `complete` marker. Before the copy, it stops all processes of user `agent`, as the end of the agent container does in condor. The copy leaves out the base model's HF repository and the agent CLI state (`.nvm`, `.npm`, `.claude`, `.codex`, `.gemini`, opencode config, and files named `oauth_token` or `api_key`), so credentials do not reach the volume. `collect.sh` logs the size and time of each phase to `/logs/artifacts/collect.log`. Without the marker, `test.sh` exits without a reward, so Harbor records an infrastructure error instead of a zero score.
4. **Judge.** `test.sh` copies only the task directory into an empty `/home/agent` and runs the Claude judge as user `agent` with a minimal environment (no `HF_TOKEN`), so it cannot change `/opt/ib`. Afterwards it stops all processes of user `agent`. It uses `src/disallowed_usage_judge/get_judge_prompt.py`, `DEFAULT_JUDGE_MODEL`, and `valid_judgements` from `judge.sh`.
5. **Evaluation.** `test.sh` restores `/home/agent` from the volume, which discards the judge's copy of the task. It removes the judge credentials from the environment. It downloads the base model with the same `snapshot_download` call as the healthcheck, before the server starts, and then removes `HF_TOKEN`. A failed hand-off or download exits without a reward, so Harbor records an infrastructure error. It starts `start_server.sh` (or `start_server_original.sh`) through `launch_supervised_server.sh` as user `agent`, with only the server, cache, and model variables that condor's server container gets. `/opt/ib` (evaluator, evaluation-seed samples, quality reference) is readable only by root, and the server sees only the launcher scripts in `/opt/inference_eval/bin`. The evaluator runs as root with a root-only `HOME` and `TMPDIR`. Before each server start, it waits for the port to close and restores the `/tmp` snapshot. It waits for readiness and runs `src/eval/tasks/<scenario>/evaluate.py` with the condor retry sequence: 3 attempts, then 2 attempts with `--request-timeout-s 150`. `valid_eval_metrics` from `src/run_task.sh` checks each attempt.

## Differences from condor

| | condor (`single_task.sub`, `run_task.sh`) | Harbor |
|---|---|---|
| GPU | 1x H100 80GB HBM3 | `gpu_types = ["H100!"]`; the `!` stops Modal from upgrading to an H200 |
| CPU and RAM | 16 CPUs, 180000 MB | `cpus = 16` (request and limit), `memory_mb = 180000` (a request, not a hard cap) |
| Disk | 400 GB | `storage_mb` is not applied by Modal |
| Agent restarts | `agents/common.sh` (claude, codex, opencode) reruns the CLI with a "continue" prompt until the budget ends; `run_task.sh` starts a new container after a crash, up to 3 runs | one Harbor agent run, limited to `num_hours` |
| HF cache | the job's cache, shared by the agent and the evaluation | the agent downloads the model, and the verifier downloads it again; other cache entries cross with `/home/agent`, but changes inside the base model's repository do not |
| Codex config | `containers/other_home_data/.codex/config.toml` | Harbor's codex setup |
| Live server at capture | the capture runs after the agent container exits and captures only a server that is still live | the capture runs before `collect.sh` stops the agent's processes, so it captures a server that the agent left running into `final_runtime_env.sh` |
| Samples in the agent image | all cached seeds | the agent seed and the quality seed only; the verifier gets the evaluation seed and the quality seed |
| Evaluation attempt timeout | none | 7200 s per attempt; verifier limit 12 h |
| Evaluation server user | the job user in an Apptainer container without credentials | `agent` in the verifier container, without credentials |
| Judge isolation | separate container with the task mounted read-only | same verifier container, as user `agent`; the task copy is discarded before the evaluation, but files that the judge writes to `/tmp` stay |
| Judge credentials | also `agents/claude_non_api/oauth_token` | only `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` |
| `nvidia-smi` wrapper, `check_cuda.py` | present | omitted |
| Starting points | `default`, `vllm_running`, `bare` | `default` only |
| Timer start | right before the agent starts | the healthcheck, before Harbor installs the agent CLI |
| Agent CLI | installed in the image | the image's CLI when the version matches (the default); otherwise Harbor installs another version in `/home/agent` |
| Container isolation | Apptainer `--cleanenv --pid`, `/tmp` bound to job scratch | Modal sandbox |
| opencode | `solve.sh` links `venv`, `.venv`, `env` in the task directory to `/tmp` | not created |

## Known limits

- Harbor 0.25 starts Modal sandboxes with `sleep infinity` and ignores the image entrypoint. The adapter does not use an entrypoint.
- The adapter targets Harbor 0.25.0.
- With the default rules, `harbor lint` reports no findings. With `--extend-select ALL`, it reports `unpinned-image`, because the base image uses a tag, as in `containers/inference.def`.
- The healthcheck and the verifier download the full model repository, as `containers/download_hf_cache/download_resources.py` does. For Mistral-7B this includes both weight formats.
- The collect hook has a 3600 s limit. A large `/home/agent` (for example a quantized model copy) can exceed it; the verifier then reports an incomplete hand-off.
- `uv` is installed in `/root/.local/bin`, which user `agent` cannot read. The Apptainer image has the same layout.
