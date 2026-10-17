"""Local Qwen3 backend with a small OpenAI-chat-compatible surface.

The travel assistant already expects ``client.chat.completions.create`` and
OpenAI-style response objects.  This adapter keeps that contract while running
Qwen3 directly with Transformers, so the same tool orchestration can be
compared against DeepSeek without an API call.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LocalFunction:
    name: str
    arguments: str


@dataclass
class LocalToolCall:
    id: str
    type: str
    function: LocalFunction


@dataclass
class LocalMessage:
    content: Optional[str]
    tool_calls: Optional[List[LocalToolCall]] = None


@dataclass
class LocalChoice:
    message: LocalMessage


@dataclass
class LocalResponse:
    choices: List[LocalChoice]


class _LocalChatCompletions:
    def __init__(self, owner: "LocalQwenClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> LocalResponse:
        return self._owner.generate(**kwargs)


class _LocalChat:
    def __init__(self, owner: "LocalQwenClient") -> None:
        self.completions = _LocalChatCompletions(owner)


class LocalQwenClient:
    """Run a local Qwen3 model behind the subset used by this project."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_new_tokens: int = 768,
        enable_thinking: bool = False,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        default_path = project_root / "models" / "qwen3-0.6b-clean"
        self.model_path = Path(model_path or os.getenv("LOCAL_QWEN_MODEL_PATH", str(default_path))).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"本地Qwen模型不存在: {self.model_path}. "
                "请先下载 Qwen/Qwen3-0.6B，或设置 LOCAL_QWEN_MODEL_PATH。"
            )

        self.max_new_tokens = int(os.getenv("LOCAL_QWEN_MAX_NEW_TOKENS", str(max_new_tokens)))
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        self.model.eval()
        self.chat = _LocalChat(self)
        self._generation_lock = threading.Lock()

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[str, List[LocalToolCall]]:
        calls: List[LocalToolCall] = []
        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for index, match in enumerate(pattern.finditer(text), start=1):
            try:
                payload = json.loads(match.group(1))
                name = payload.get("name")
                arguments = payload.get("arguments", {})
                if not isinstance(name, str) or not name:
                    continue
                if isinstance(arguments, str):
                    arguments_text = arguments
                else:
                    arguments_text = json.dumps(arguments, ensure_ascii=False)
                calls.append(
                    LocalToolCall(
                        id=f"call_local_{index}",
                        type="function",
                        function=LocalFunction(name=name, arguments=arguments_text),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

        clean = pattern.sub("", text)
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
        clean = clean.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return clean, calls

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
        **_: Any,
    ) -> LocalResponse:
        template_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "enable_thinking": self.enable_thinking,
        }
        if tools:
            template_kwargs["tools"] = tools

        encoded = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(encoded, dict):
            inputs = {key: value.to(self._device) for key, value in encoded.items()}
            input_length = inputs["input_ids"].shape[-1]
        else:
            input_ids = encoded.to(self._device)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
            input_length = input_ids.shape[-1]

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens or self.max_new_tokens),
            "do_sample": False,
            # Qwen's saved generation config contains sampling defaults. Clear
            # them explicitly for deterministic comparison runs.
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # Transformers generation is not safe to run concurrently on the same
        # model. A shared client is used by the data-conversion utilities, so
        # serialize generation while still reusing one loaded model.
        with self._generation_lock, torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, input_length:]
        raw_text = self.tokenizer.decode(generated, skip_special_tokens=False)
        content, tool_calls = self._parse_tool_calls(raw_text)
        message = LocalMessage(content=content or None, tool_calls=tool_calls or None)
        return LocalResponse(choices=[LocalChoice(message=message)])


_CLIENTS: Dict[str, LocalQwenClient] = {}
_CLIENTS_LOCK = threading.Lock()


def get_local_qwen_client(model_path: Optional[str] = None) -> LocalQwenClient:
    """Return one process-local model instance for a model path."""
    key = str(Path(model_path or os.getenv("LOCAL_QWEN_MODEL_PATH", "")).expanduser())
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            _CLIENTS[key] = LocalQwenClient(model_path=model_path)
        return _CLIENTS[key]
