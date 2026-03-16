from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from anthropic import Anthropic


def _extract_system_and_user(messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    あなたの builder が作る messages（[system, user]）から
    system文字列と userメッセージ(dict)を取り出す。
    """
    system_text = ""
    user_msg: Optional[Dict[str, Any]] = None

    for m in messages:
        role = m.get("role")
        if role == "system":
            # OpenAIではcontentが文字列
            system_text = m.get("content", "") or ""
        elif role == "user":
            user_msg = m

    if user_msg is None:
        raise ValueError("No user message found in messages")

    return system_text, user_msg


def _parse_data_url(data_url: str) -> Tuple[str, str]:
    """
    data:<media_type>;base64,<data> を (media_type, base64_data) に分解する。
    """
    if not data_url.startswith("data:"):
        raise ValueError("Not a data URL")

    # data:image/png;base64,XXXX
    header, b64 = data_url.split(",", 1)
    meta = header[len("data:") :]  # image/png;base64
    parts = meta.split(";")
    media_type = parts[0] if parts else "application/octet-stream"
    # base64の指定が無いケースもあり得るが、builderは必ずbase64
    return media_type, b64


def _openai_user_content_to_claude_blocks(user_content: Any) -> List[Dict[str, Any]]:
    """
    OpenAIの user.content（list of {type,text} / {type:image_url,...}）を
    Claude Messages の content blocks に変換する。

    Claude Vision は base64 / url source をサポート :contentReference[oaicite:4]{index=4}
    """
    if isinstance(user_content, str):
        return [{"type": "text", "text": user_content}]

    if not isinstance(user_content, list):
        raise ValueError("Unsupported user content type")

    blocks: List[Dict[str, Any]] = []
    for part in user_content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")

        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "") or ""})
            continue

        if ptype == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if not url:
                continue

            if url.startswith("data:"):
                media_type, b64 = _parse_data_url(url)
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    }
                )
            else:
                # URL source も Claude 側でサポートされる :contentReference[oaicite:5]{index=5}
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    }
                )
            continue

        # unknown part type: ignore
    return blocks


class ClaudeClient:
    """
    Anthropic Claude API クライアント。

    - 環境変数 ANTHROPIC_API_KEY を利用（SDKのデフォルトでも読み込む） :contentReference[oaicite:6]{index=6}
    - OpenAI形式 messages を Claude Messages API 形式へ変換して投げる
    - run(messages) -> str（OpenAIClient互換）
    """

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key is not provided (env: {api_key_env})")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def run(self, messages: List[Dict[str, Any]]) -> str:
        system_text, user_msg = _extract_system_and_user(messages)

        user_blocks = _openai_user_content_to_claude_blocks(user_msg.get("content"))

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_text or None,
            messages=[
                {
                    "role": "user",
                    "content": user_blocks,
                }
            ],
            # temperature=0.0,
        )

        # resp.content は content blocks のリスト（text等）
        texts: List[str] = []
        for block in getattr(resp, "content", []) or []:
            # SDKのblockは dictライク / pydantic の場合があるので両対応
            btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if btype == "text":
                t = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                if t:
                    texts.append(t)

        return "\n".join(texts).strip()
