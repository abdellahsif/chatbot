from __future__ import annotations

import json
import re
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models import UserProfile


class QwenGenerator:
    def __init__(self) -> None:
        self._lock = Lock()
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._tokenizer is not None and self._model is not None

        with self._lock:
            if self._loaded:
                return self._tokenizer is not None and self._model is not None
            model_id = "Qwen/Qwen2.5-3B-Instruct"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                self._model.eval()
            except Exception:
                self._tokenizer = None
                self._model = None
            self._loaded = True
            return self._tokenizer is not None and self._model is not None

    @staticmethod
    def _extract_json_block(text: str) -> dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    def generate(
        self,
        *,
        question: str,
        profile: UserProfile,
        hits: list[dict],
    ) -> dict[str, str]:
        if not hits:
            return {
                "short_answer": "I do not have enough matching data for your current profile and constraints.",
                "why_it_fits": "Your constraints removed all candidates from retrieval.",
                "alternative": "Try widening your budget or changing city.",
                "next_action": "Tell me your target program and city.",
            }

        top_hits = hits[:5]
        evidence_lines = []
        for item in top_hits:
            c = item["chunk"]
            s = item["school"]
            evidence_lines.append(
                f"chunk_id={c.get('chunk_id', '')} | school={s.get('name', '')} | city={s.get('city', '')} | "
                f"program={c.get('program', '')} | tuition_max={s.get('tuition_max_mad', '')} | text={str(c.get('text', ''))[:500]}"
            )

        prompt = (
            "You are a strict education advisor. Use only provided evidence. Do not invent facts. "
            "If data is missing, say it clearly and ask one focused follow-up in next_action.\n\n"
            f"Question: {question}\n"
            f"Profile: bac_stream={profile.bac_stream}, expected_grade_band={profile.expected_grade_band}, "
            f"motivation={profile.motivation}, budget_band={profile.budget_band}, city={profile.city}, country={profile.country}\n\n"
            "Evidence:\n"
            + "\n".join(f"- {line}" for line in evidence_lines)
            + "\n\nReturn ONLY valid JSON with keys: short_answer, why_it_fits, alternative, next_action"
        )

        if not self._ensure_loaded():
            top_school = top_hits[0]["school"]
            return {
                "short_answer": f"Top match right now is {top_school.get('name', 'the top school')}.",
                "why_it_fits": "This result is based on semantic retrieval against your profile constraints.",
                "alternative": "Ask me to compare two schools for a direct trade-off.",
                "next_action": "Tell me the exact program you want so I can narrow options.",
            }

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = self._model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=220,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            raw = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = self._extract_json_block(raw)
        except Exception:
            parsed = {}

        top_school = top_hits[0]["school"]
        return {
            "short_answer": str(parsed.get("short_answer", "")).strip()
            or f"Top match right now is {top_school.get('name', 'the top school')}",
            "why_it_fits": str(parsed.get("why_it_fits", "")).strip()
            or "This recommendation aligns with your profile constraints and retrieved evidence.",
            "alternative": str(parsed.get("alternative", "")).strip()
            or "Ask for one alternative with a lower cost or different city.",
            "next_action": str(parsed.get("next_action", "")).strip()
            or "Tell me your exact target program and city to refine results.",
        }


QWEN_GENERATOR = QwenGenerator()
