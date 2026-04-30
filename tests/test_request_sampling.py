import json

import pytest

from src.eval.inference import runner


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        tokens = []
        for message in messages:
            content = message.get("content", "")
            if content:
                tokens.extend(content.split())
        return tokens

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return " ".join(tokens)


def _write_longbench_samples(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _speed_config(num_requests=12):
    return {
        "dataset_seed": 123,
        "num_requests": num_requests,
        "temperature": 0.2,
        "synthetic": {
            "input_len": 10,
            "output_len": 5,
            "range_ratio": 0.8,
            "ignore_eos": True,
        },
    }


def test_prepare_requests_samples_input_targets_with_replacement(tmp_path, monkeypatch):
    samples_path = tmp_path / "samples.jsonl"
    _write_longbench_samples(
        samples_path,
        [
            {
                "sample_id": "only-source",
                "messages": [{"role": "user", "content": " ".join(f"tok{i}" for i in range(20))}],
            }
        ],
    )
    monkeypatch.setattr(runner, "_find_longbench_samples_file", lambda *args, **kwargs: samples_path)

    requests, samples = runner._prepare_requests(
        _speed_config(),
        limit=None,
        tokenizer=FakeTokenizer(),
        max_model_len=None,
    )

    assert len(requests) == 12
    assert len(samples) == 12
    assert {req["sample_id"] for req in requests} == {"only-source"}
    targets = [req["target_input_token_count"] for req in requests]
    assert all(8 <= target <= 10 for target in targets)
    assert len(set(targets)) > 1
    assert all(req["input_token_count"] == req["target_input_token_count"] for req in requests)
    assert all(req["source_prompt_token_count"] == 20 for req in requests)
    assert all(4 <= req["max_new_tokens"] <= 5 for req in requests)
    assert all(req["sampling_range_ratio"] == 0.8 for req in requests)


def test_prepare_requests_requires_tokenizer_for_implicit_sampling(tmp_path, monkeypatch):
    samples_path = tmp_path / "samples.jsonl"
    _write_longbench_samples(
        samples_path,
        [{"sample_id": "sample", "messages": [{"role": "user", "content": "tok0 tok1 tok2"}]}],
    )
    monkeypatch.setattr(runner, "_find_longbench_samples_file", lambda *args, **kwargs: samples_path)

    with pytest.raises(RuntimeError, match="requires the base model tokenizer"):
        runner._prepare_requests(_speed_config(num_requests=1), limit=None, tokenizer=None, max_model_len=None)
