from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI


class OpenAIClient:
    def __init__(self, model: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not provided")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0.0,
        )
        # 1.x 系では attributes/object なので
        return resp.choices[0].message.content
