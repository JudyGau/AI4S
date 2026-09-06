"""LLM Provider 抽象与 OpenAI 兼容实现（仅用标准库 urllib，无第三方依赖）。

OpenAI 兼容的 `/chat/completions` 即可覆盖 DeepSeek / 通义 / 本地 vLLM 等，
只需更换 `base_url` 与 `model`。
"""
from __future__ import annotations

import json
import urllib.request


class LLMError(Exception):
    """LLM 调用层错误（配置缺失、网络/协议返回异常等）。"""


class LLMConfigError(LLMError):
    """LLM 配置缺失或非法。"""


class LlmProvider:
    """统一聊天接口抽象。"""

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 256) -> str:
        raise NotImplementedError


class OpenAiCompatible(LlmProvider):
    """通过 OpenAI 兼容 `/chat/completions` 调用任意模型。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 30.0, extra_body: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 追加到请求体的额外字段（如 GLM 的思考强度 {thinking/type/budget_tokens} 等厂商专属参数）
        self.extra_body = extra_body or {}

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 256) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_body)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except LLMError:
            raise
        except Exception as exc:  # 网络/HTTP/JSON 异常统一包装
            raise LLMError(f"调用 LLM 失败: {exc}") from exc
        try:
            msg = data["choices"][0].get("message", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 返回结构异常: {data}") from exc
        content = msg.get("content") or ""
        # 推理模型可能只输出 reasoning_content（思考链）而没有干净的 content，
        # 且少数模型会把最终正文一并放进 reasoning_content —— 因此正文为空时回退思考链，
        # 让上层解析器决定能否从中取出表达式。
        if not content:
            content = msg.get("reasoning_content") or ""
        return content