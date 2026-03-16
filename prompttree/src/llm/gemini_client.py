from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


class GeminiClient:
    """
    Gemini API (Developer API) を OpenAI互換エンドポイント経由で叩くクライアント。

    - OpenAI Python SDK をそのまま利用
    - base_url を Gemini の OpenAI互換に差し替える
    - 環境変数 GOOGLE_API_KEY を利用

    参考: Gemini OpenAI compatibility の公式ドキュメント :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, model: str, api_key_env: str = "GOOGLE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key is not provided (env: {api_key_env})")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.model = model

    def run(self, messages: List[Dict[str, Any]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0.0,
        )
        # OpenAI SDK互換の返り値
        return resp.choices[0].message.content
