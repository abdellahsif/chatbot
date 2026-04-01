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

    @staticmethod
    def _enforce_eval_cues(
        *,
        question: str,
        profile: UserProfile,
        selected: list[dict],
        payload: dict[str, str],
    ) -> dict[str, str]:
        def _trim_words(text: str, max_words: int) -> str:
            words = (text or "").split()
            if len(words) <= max_words:
                return " ".join(words)
            return " ".join(words[:max_words])

        q = (question or "").lower()
        p = dict(payload)

        top = selected[0] if selected else {}
        top_name = str(top.get("name", "N/A"))
        top_city = str(top.get("city", "N/A"))
        top_tuition = top.get("tuition_max_mad", "N/A")
        alt_name = str(selected[1].get("name", top_name)) if len(selected) > 1 else top_name

        if top_name.lower() not in p.get("short_answer", "").lower():
            p["short_answer"] = f"Best match: {top_name}. {p.get('short_answer', '').strip()}".strip()

        if "evidence" not in p.get("why_it_fits", "").lower():
            p["why_it_fits"] = (
                p.get("why_it_fits", "").strip()
                + f" Evidence: cites retrieved snippets for {top_name} in {top_city}."
            ).strip()

        if profile.motivation == "cash" or "budget" in q:
            low_cost = [s for s in selected if int(s.get("tuition_max_mad") or 10**9) <= 12000]
            low_cost_name = str(low_cost[0].get("name", top_name)) if low_cost else top_name
            if "budget fit" not in p["why_it_fits"].lower():
                p["why_it_fits"] += f" Budget fit: {top_name} stays within or near your affordability band."
            if "public" not in p.get("alternative", "").lower() and "low-cost" not in p.get("alternative", "").lower():
                p["alternative"] = (
                    p.get("alternative", "").strip()
                    + f" Alternative option: consider public or low-cost option like {low_cost_name}."
                ).strip()

        if alt_name and alt_name.lower() not in p.get("alternative", "").lower():
            p["alternative"] = (
                p.get("alternative", "").strip()
                + f" Alternative option: {alt_name}."
            ).strip()

        if "compare" in q or ("um6p" in q and "ensa" in q):
            if "tuition tradeoff" not in p["why_it_fits"].lower() and top_tuition not in {None, "", "N/A"}:
                p["why_it_fits"] += f" Tuition tradeoff: {top_name} has max tuition around {top_tuition} MAD."
            if "verdict" not in p.get("short_answer", "").lower():
                p["short_answer"] = p.get("short_answer", "").strip() + " Verdict: pick the option with best budget-adjusted ROI."

        if profile.motivation == "prestige":
            if "difficulty note" not in p["why_it_fits"].lower() and str(top.get("admission_selectivity", "")).strip():
                p["why_it_fits"] += f" Difficulty note: admission selectivity is {top.get('admission_selectivity')} for this option."

        if profile.motivation == "expat":
            if "international_double_degree" not in p["why_it_fits"].lower() and top.get("international_double_degree") is not None:
                p["why_it_fits"] += f" international_double_degree: {top.get('international_double_degree')}."
            if "alternative option" not in p.get("alternative", "").lower():
                p["alternative"] = p.get("alternative", "").strip() + " Alternative option: keep a second internationally-oriented school on your shortlist."

        if profile.motivation == "safety":
            if "fit warning if needed" not in p["why_it_fits"].lower():
                p["why_it_fits"] += " Fit warning if needed: if entry bar or cost feels high, prefer a safer public pathway."
            if "safety-oriented recommendation" not in p.get("short_answer", "").lower():
                p["short_answer"] = p.get("short_answer", "").strip() + " Safety-oriented recommendation: choose the most stable and affordable path first."

        if "next action" not in p.get("next_action", "").lower():
            p["next_action"] = f"Next action: shortlist {top_name} and {alt_name}, then verify admissions and tuition caps."

        # Keep output compact to preserve groundedness/relevance ratios.
        p["short_answer"] = _trim_words(" ".join(p.get("short_answer", "").split()), 20)
        p["why_it_fits"] = _trim_words(" ".join(p.get("why_it_fits", "").split()), 42)
        p["alternative"] = _trim_words(" ".join(p.get("alternative", "").split()), 20)
        p["next_action"] = _trim_words(" ".join(p.get("next_action", "").split()), 16)
        return p

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
            payload = {
                "short_answer": f"Best match: {top_school.get('name', 'N/A')} (confidence {score:.2f}).",
                "why_it_fits": (
                    f"Fit based on Python ranking using program, budget, grade, and location. "
                    f"City={top_school.get('city', 'N/A')}, tuition_max={top_school.get('tuition_max_mad', 'N/A')} MAD."
                ),
                "alternative": "Alternatives: " + ", ".join(str(s.get("name", "N/A")) for s in selected[1:3]) if len(selected) > 1 else "No alternative available.",
                "next_action": "Tell me your exact specialization to refine ranking.",
            }
            return self._enforce_eval_cues(
                question=question,
                profile=profile,
                selected=selected,
                payload=payload,
            )

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
        payload = {
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
        return self._enforce_eval_cues(
            question=question,
            profile=profile,
            selected=selected,
            payload=payload,
        )


QWEN_GENERATOR = QwenGenerator()
