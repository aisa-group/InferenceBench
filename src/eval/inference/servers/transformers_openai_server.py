#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from aiohttp import web
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def _messages_to_prompt(messages: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, torch.Tensor]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(input_ids, "keys"):
                input_ids = input_ids["input_ids"]
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        except Exception:
            pass
    joined = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)
    out = tokenizer(joined, return_tensors="pt")
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(out["input_ids"])
    return out


def _extract_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    msgs = payload.get("messages") or []
    if not isinstance(msgs, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _now() -> int:
    return int(time.time())


@dataclass
class ServerState:
    model_id: str
    model: Any
    tokenizer: Any
    max_model_len: Optional[int]


async def handle_models(request: web.Request) -> web.Response:
    state: ServerState = request.app["state"]
    return web.json_response({"object": "list", "data": [{"id": state.model_id, "object": "model"}]})


def _prepare_inputs(
    state: ServerState,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
) -> Dict[str, torch.Tensor]:
    inputs = _messages_to_prompt(messages, state.tokenizer)
    if state.max_model_len is not None:
        max_input_tokens = state.max_model_len - max_new_tokens - 1
        if max_input_tokens <= 0:
            raise ValueError("max_tokens leaves no room for input tokens")
        if inputs["input_ids"].shape[1] > max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]
    return {k: v.to(state.model.device) for k, v in inputs.items()}


class _GenerationStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer: Any):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.output_tokens = 0
        self.error: Optional[str] = None
        self.cancelled = threading.Event()

    def put(self, value) -> None:
        if not self.next_tokens_are_prompt:
            self.output_tokens += value.numel()
        super().put(value)


def _start_generation(
    state: ServerState,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    ignore_eos: bool = False,
) -> Tuple[_GenerationStreamer, threading.Thread]:
    inputs = _prepare_inputs(state, messages, max_new_tokens)
    do_sample = temperature > 0.0
    streamer = _GenerationStreamer(state.tokenizer)

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": state.tokenizer.eos_token_id,
        "streamer": streamer,
        "stopping_criteria": [lambda input_ids, scores, **kwargs: streamer.cancelled.is_set()],
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    if ignore_eos:
        gen_kwargs["eos_token_id"] = None
        gen_kwargs["forced_eos_token_id"] = None

    def _run() -> None:
        try:
            with torch.inference_mode():
                state.model.generate(**gen_kwargs)
        except Exception as exc:
            print(f"[torch-server] generate() raised: {exc!r}", flush=True)
            streamer.error = str(exc)
            streamer.end()

    thread = threading.Thread(target=_run, name="torch-generate", daemon=True)
    thread.start()
    return streamer, thread


def _chunk_queue(
    streamer: TextIteratorStreamer,
) -> "asyncio.Queue[Optional[str]]":
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _pump() -> None:
        try:
            for piece in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, piece)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_pump, name="torch-stream-pump", daemon=True).start()
    return queue


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    state: ServerState = request.app["state"]
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 256)))
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        if max_tokens <= 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("max_tokens must be positive and temperature must be finite and nonnegative")
    except (ValueError, TypeError) as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=400)

    messages = _extract_messages(payload)
    if not messages:
        return web.json_response({"error": {"message": "missing messages"}}, status=400)

    stream = bool(payload.get("stream", True))
    try:
        streamer, gen_thread = _start_generation(
            state,
            messages,
            max_new_tokens=max_tokens,
            temperature=temperature,
            ignore_eos=bool(payload.get("ignore_eos", False)),
        )
    except ValueError as exc:
        return web.json_response({"error": {"message": str(exc)}}, status=400)
    try:
        return await _completion_response(request, state.model_id, stream, streamer, gen_thread)
    finally:
        streamer.cancelled.set()
        await asyncio.to_thread(gen_thread.join)


async def _completion_response(
    request: web.Request,
    model_id: str,
    stream: bool,
    streamer: _GenerationStreamer,
    gen_thread: threading.Thread,
) -> web.StreamResponse:
    chunk_queue = _chunk_queue(streamer)
    chat_id = f"chatcmpl-{_now()}"

    if not stream:
        text_parts: List[str] = []
        while True:
            piece = await chunk_queue.get()
            if piece is None:
                break
            if piece:
                text_parts.append(piece)
        await asyncio.to_thread(gen_thread.join)
        if streamer.error is not None:
            return web.json_response({"error": {"message": streamer.error}}, status=500)
        response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": _now(),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": streamer.output_tokens},
        }
        return web.json_response(response)

    resp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    while True:
        piece = await chunk_queue.get()
        if piece is None:
            break
        if not piece:
            continue
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _now(),
            "model": model_id,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        await resp.write(
            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        )

    await asyncio.to_thread(gen_thread.join)
    if streamer.error is not None:
        await resp.write(f"data: {json.dumps({'error': {'message': streamer.error}})}\n\n".encode("utf-8"))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": streamer.output_tokens},
    }
    await resp.write(
        f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
    )
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible server using Transformers generate().")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "0") or 0),
        help="Context length used to truncate inputs; 0 reads it from the model config.",
    )
    args = parser.parse_args()

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype), device_map=device_map)
    model.eval()

    max_model_len: Optional[int] = None
    if args.max_model_len > 0:
        max_model_len = args.max_model_len
    else:
        cfg_len = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "model_max_length", None)
        if isinstance(cfg_len, int) and cfg_len > 0:
            max_model_len = cfg_len

    app = web.Application()
    app["state"] = ServerState(model_id=args.model, model=model, tokenizer=tokenizer, max_model_len=max_model_len)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)

    web.run_app(app, host=args.host, port=args.port, handler_cancellation=True)


if __name__ == "__main__":
    main()
