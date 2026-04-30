#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
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


class ServerState:
    def __init__(self, model_id: str, model: Any, tokenizer: Any, max_model_len: Optional[int]):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len


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
        max_input_tokens = max(0, int(state.max_model_len) - int(max_new_tokens) - 1)
        if max_input_tokens > 0 and int(inputs["input_ids"].shape[1]) > max_input_tokens:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_tokens:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]
    return {k: v.to(state.model.device) for k, v in inputs.items()}


def _start_generation(
    state: ServerState,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
) -> Tuple[TextIteratorStreamer, threading.Thread]:
    """Launch `model.generate()` in a background thread and return its streamer.

    The streamer yields decoded text chunks from the foreground; each yielded
    chunk corresponds to one decode step of the model. This gives clients a
    real first-token time and a real inter-token latency signal, unlike the
    previous fake-streaming path that ran `generate()` to completion and then
    iterated the finished token list.
    """
    inputs = _prepare_inputs(state, messages, max_new_tokens)
    do_sample = temperature is not None and float(temperature) > 0.0

    streamer = TextIteratorStreamer(
        state.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": state.tokenizer.eos_token_id,
        "streamer": streamer,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    def _run() -> None:
        try:
            with torch.inference_mode():
                state.model.generate(**gen_kwargs)
        except Exception as exc:
            # The streamer has no error channel; log and let the iterator end.
            print(f"[torch-server] generate() raised: {exc!r}", flush=True)

    thread = threading.Thread(target=_run, name="torch-generate", daemon=True)
    thread.start()
    return streamer, thread


async def _astream_chunks(
    streamer: TextIteratorStreamer,
) -> "asyncio.Queue[Optional[str]]":
    """Bridge the blocking streamer into an asyncio queue of text chunks.

    `None` is pushed once as the terminal sentinel.
    """
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
    payload = await request.json()

    messages = _extract_messages(payload)
    if not messages:
        return web.json_response({"error": {"message": "missing messages"}}, status=400)

    stream = bool(payload.get("stream", True))
    max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 256)))
    temperature = float(payload.get("temperature", 0.0) or 0.0)

    streamer, gen_thread = _start_generation(
        state,
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    chunk_queue = await _astream_chunks(streamer)
    chat_id = f"chatcmpl-{_now()}"

    if not stream:
        # Non-streaming path: still drive generation via the same streamer so
        # that generate() isn't invoked twice. Collect all chunks, then reply.
        text_parts: List[str] = []
        chunk_count = 0
        while True:
            piece = await chunk_queue.get()
            if piece is None:
                break
            if piece:
                text_parts.append(piece)
                chunk_count += 1
        await asyncio.to_thread(gen_thread.join)
        response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": _now(),
            "model": state.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(text_parts)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": chunk_count},
        }
        return web.json_response(response)

    # Streaming path: flush each decoded chunk to the client as soon as the
    # background generation thread produces it. Each yielded chunk corresponds
    # to one decode step, which is the per-token granularity we want for TTFT,
    # ITL, and TPOT measurement.
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

    chunk_count = 0
    while True:
        piece = await chunk_queue.get()
        if piece is None:
            break
        if not piece:
            # Empty string from the streamer (e.g., partial multi-byte buffering)
            # — skip without emitting an SSE chunk, so we don't inflate token
            # counts with no visible text.
            continue
        chunk_count += 1
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": _now(),
            "model": state.model_id,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        await resp.write(
            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        )

    await asyncio.to_thread(gen_thread.join)

    # Final chunk with finish_reason and usage.
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": state.model_id,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": chunk_count},
    }
    await resp.write(
        f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
    )
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic OpenAI-compatible server using Transformers generate().")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8001")))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_MAX_MODEL_LEN", "0") or 0),
        help="If set >0, truncate inputs to fit max_model_len - max_new_tokens - 1.",
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16
    device_map = "cuda" if use_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, device_map=device_map)
    model.eval()

    max_model_len: Optional[int] = None
    if args.max_model_len and int(args.max_model_len) > 0:
        max_model_len = int(args.max_model_len)
    else:
        try:
            cfg_len = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "model_max_length", None)
            if isinstance(cfg_len, int) and cfg_len > 0:
                max_model_len = cfg_len
        except Exception:
            max_model_len = None

    app = web.Application()
    app["state"] = ServerState(model_id=args.model, model=model, tokenizer=tokenizer, max_model_len=max_model_len)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
