import json
import re
from typing import Any, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PredictedStructure:
    reading_order: List[int]
    tree: List[Dict[str, int]]  # {"bbox_number": int, "parent": int}


def parse_llm_output(text: str) -> PredictedStructure:
    obj = json.loads(text)
    return PredictedStructure(
        reading_order=obj["reading_order"],
        tree=obj["tree"],
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def try_parse_llm_output(text: str) -> Optional[PredictedStructure]:
    """
    失敗しても例外を投げずに None を返す。
    最低限のスキーマ検証も行う（reading_order / tree の存在と型）。
    """
    def _from_obj(obj: Any) -> Optional[PredictedStructure]:
        if not isinstance(obj, dict):
            return None
        if "reading_order" not in obj or "tree" not in obj:
            return None
        if not isinstance(obj["reading_order"], list):
            return None
        if not isinstance(obj["tree"], list):
            return None
        for rel in obj["tree"]:
            if not isinstance(rel, dict):
                return None
            if "bbox_number" not in rel or "parent" not in rel:
                return None
        return PredictedStructure(
            reading_order=obj["reading_order"],
            tree=obj["tree"],
        )

    try:
        return _from_obj(json.loads(text))
    except Exception:
        pass

    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None

    try:
        return _from_obj(json.loads(m.group(0)))
    except Exception:
        return None
