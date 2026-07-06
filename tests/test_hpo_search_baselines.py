from src.eval.inference import hpo_search_baselines as hpo


def test_search_spaces_are_loadable():
    for engine, expected_count in {"vllm": 11, "sglang": 10, "tgi": 9}.items():
        raw, params = hpo.load_search_space(engine)
        assert raw["engine"] == engine
        assert len(params) == expected_count


def test_vllm_command_renders_special_flags():
    _, params = hpo.load_search_space("vllm")
    config = {
        "max_num_seqs": 16,
        "max_num_batched_tokens": 1024,
        "gpu_memory_utilization": 0.9,
        "block_size": 16,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "quantization": "none",
        "kv_cache_dtype": "fp8",
        "attention_backend": "FLASHINFER",
        "num_speculative_tokens": 5,
    }
    cmd = hpo.build_engine_cmd(
        "vllm",
        config,
        params,
        "mistralai/Mistral-7B-Instruct-v0.3",
        "127.0.0.1",
        8000,
        "32768",
        21,
    )
    joined = " ".join(cmd)
    assert "--no-enable-chunked-prefill" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--quantization" not in cmd
    assert "--kv-cache-dtype fp8" in joined
    assert "--attention-backend" not in cmd
    assert "--speculative-config" in cmd
    assert '"method":"ngram"' in cmd[cmd.index("--speculative-config") + 1]

    env = hpo.build_trial_env("vllm", config, {}, skip_quality=True)
    assert env["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"


def test_sglang_search_space_is_stable_by_default(monkeypatch):
    monkeypatch.setattr(hpo, "_load_installed_sglang_cli_spec", lambda: {})
    _, params = hpo.load_search_space("sglang")

    hpo.validate_search_space("sglang", params)


def test_sglang_search_space_rejects_unsafe_defaults(monkeypatch):
    monkeypatch.setattr(hpo, "_load_installed_sglang_cli_spec", lambda: {})
    params = [
        hpo.SearchParam(
            name="quantization",
            flag="--quantization",
            kind="choice",
            values=["none", "awq"],
        )
    ]

    try:
        hpo.validate_search_space("sglang", params)
    except ValueError as exc:
        assert "quantization" in str(exc)
        assert "unsafe values" in str(exc)
    else:
        raise AssertionError("unsafe SGLang search space should fail validation")


def test_sglang_command_renders_safe_flags():
    _, params = hpo.load_search_space("sglang")
    config = {param.name: param.values[0] for param in params}

    cmd = hpo.build_engine_cmd(
        "sglang",
        config,
        params,
        "mistralai/Mistral-7B-Instruct-v0.3",
        "127.0.0.1",
        8000,
        "32768",
        21,
    )

    assert "--attention-backend" in cmd
    assert cmd[cmd.index("--attention-backend") + 1] == "triton"
    assert "--disable-cuda-graph" in cmd
    assert "--disable-piecewise-cuda-graph" in cmd
    assert "--quantization" not in cmd
    assert "--enable-torch-compile" not in cmd
    assert "--speculative-algorithm" not in cmd


def test_primary_metric_uses_profile_objectives():
    metrics = {
        "profiles": {
            "burst": {
                "ttft": {"p50": 0.25},
                "tpot": {"p50": 0.01},
                "request_throughput_req_per_s": 8.0,
                "generation_throughput_tokens_per_s": 1000.0,
            },
            "poisson": {
                "ttft": {"p50": 0.30},
                "tpot": {"p50": 0.02},
                "request_throughput_req_per_s": 2.0,
                "generation_throughput_tokens_per_s": 900.0,
            },
            "constant": {
                "ttft": {"p50": 0.35},
                "tpot": {"p50": 0.03},
                "request_throughput_req_per_s": 4.0,
                "generation_throughput_tokens_per_s": 800.0,
            },
        }
    }
    metric, source = hpo.primary_metric(metrics, "inference_scenario_a_input_heavy")
    assert metric == 4.0
    assert source == "1/ttft_p50"

    metric, source = hpo.primary_metric(metrics, "inference_scenario_c_high_load")
    assert metric == (8.0 * 2.0 * 4.0) ** (1.0 / 3.0)
    assert source == "geomean_request_throughput_req_per_s"


def test_gate_passed_ignores_legacy_success_gate_field():
    legacy_key = "success_rate" + "_check"
    metrics = {
        "quality_check": {"pass": True},
        legacy_key: {"pass": False},
    }

    assert hpo.gate_passed(metrics)


def test_gate_passed_still_fails_quality_and_zero_score():
    assert not hpo.gate_passed({"score": 0, "quality_check": {"pass": True}})
    assert not hpo.gate_passed({"quality_check": {"pass": False}})


def test_gate_passed_requires_explicit_successful_quality_check():
    assert hpo.gate_passed({"quality_check": {"pass": True}})
    assert not hpo.gate_passed({})
    assert not hpo.gate_passed({"quality_check": {}})
    assert not hpo.gate_passed(
        {
            "error": "evaluate.py exited with code 1",
            "quality_check": {"pass": True},
        }
    )
