from __future__ import annotations

import json
import os
import re
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models import UserProfile


class QwenGenerator:
    def __init__(self, model_id: str | None = None) -> None:
        self._lock = Lock()
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self.model_id = model_id or os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._tokenizer is not None and self._model is not None

        with self._lock:
            if self._loaded:
                return self._tokenizer is not None and self._model is not None
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
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
        top_schools: list[dict],
    ) -> dict[str, str]:
        if not top_schools:
            return {
                "short_answer": "No suitable school found for your constraints.",
                "why_it_fits": "No candidate passed strict filtering on budget/bac/country.",
                "alternative": "Try widening budget, changing city, or selecting a related program.",
                "next_action": "Tell me your exact program and whether your budget can be extended.",
            }

        selected = top_schools[:5]
        selected_json = json.dumps(selected, ensure_ascii=True)

        prompt = (
            "You are a strict education recommendation explainer. "
            "School selection is already done by Python. You must NOT change ranking and must NOT add any external school. "
            "Use ONLY provided JSON schools. If empty, say 'No suitable school found...'.\n\n"
            f"Question: {question}\n"
            f"Profile: bac_stream={profile.bac_stream}, expected_grade_band={profile.expected_grade_band}, "
            f"motivation={profile.motivation}, budget_band={profile.budget_band}, city={profile.city}, country={profile.country}\n\n"
            f"Selected schools JSON (ordered best to worst):\n{selected_json}\n\n"
            "Return ONLY valid JSON with keys: short_answer, why_it_fits, alternative, next_action. "
            "Requirements: short_answer mentions Best match and confidence score (0-1). "
            "why_it_fits must reference concrete attributes: city, tuition, and score components. "
            "alternative must include 1-2 school names from the provided JSON."
        )

        if not self._ensure_loaded():
            top_school = selected[0]
            score = float(top_school.get("score", 0.0))
            return {
                "short_answer": f"Best match: {top_school.get('name', 'N/A')} (confidence {score:.2f}).",
                "why_it_fits": (
                    f"Fit based on Python ranking using program, budget, grade, and location. "
                    f"City={top_school.get('city', 'N/A')}, tuition_max={top_school.get('tuition_max_mad', 'N/A')} MAD."
                ),
                "alternative": "Alternatives: " + ", ".join(str(s.get("name", "N/A")) for s in selected[1:3]) if len(selected) > 1 else "No alternative available.",
                "next_action": "Tell me your exact specialization to refine ranking.",
            }

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            device = self._model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            raw = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = self._extract_json_block(raw)
        except Exception:
            parsed = {}

        top_school = selected[0]
        score = float(top_school.get("score", 0.0))
        alternatives = ", ".join(str(s.get("name", "N/A")) for s in selected[1:3]) if len(selected) > 1 else "No alternative available"
        return {
            "short_answer": str(parsed.get("short_answer", "")).strip()
            or f"Best match: {top_school.get('name', 'N/A')} (confidence {score:.2f}).",
            "why_it_fits": str(parsed.get("why_it_fits", "")).strip()
            or (
                f"This recommendation matches your budget and profile fit. "
                f"City={top_school.get('city', 'N/A')}, tuition_max={top_school.get('tuition_max_mad', 'N/A')} MAD, "
                f"weighted_score={score:.2f}."
            ),
            "alternative": str(parsed.get("alternative", "")).strip()
            or f"Alternatives: {alternatives}.",
            "next_action": str(parsed.get("next_action", "")).strip()
            or "Share your exact program and acceptable tuition cap to refine ranking.",
        }


QWEN_GENERATOR = QwenGenerator()
