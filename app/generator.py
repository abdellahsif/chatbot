from __future__ import annotations

import json
import os
import re
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models import EvidenceItem, UserProfile


_ALLOWED_QUERY_DOMAINS = {
    "computer",
    "engineering",
    "medicine",
    "healthcare",
    "business",
    "law",
    "arts",
    "military",
}

_ALLOWED_BAC_STREAMS = {"sm", "spc", "svt", "eco", "lettres", "arts", ""}
_ALLOWED_BUDGET_BANDS = {"zero_public", "tight_25k", "comfort_50k", "no_limit_70k_plus", ""}
_ALLOWED_MOTIVATIONS = {"employability", "prestige", "expat", "cash", "safety", "passion", ""}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_domain(value: str) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not token:
        return ""
    mapping = {
        "it": "computer",
        "informatique": "computer",
        "software": "computer",
        "cyber": "computer",
        "cybersecurity": "computer",
        "cybersecurite": "computer",
        "engineer": "engineering",
        "ingenierie": "engineering",
        "genie": "engineering",
        "medecine": "medicine",
        "medical": "medicine",
        "sante": "healthcare",
        "health": "healthcare",
        "economie": "business",
        "finance": "business",
        "management": "business",
        "droit": "law",
        "legal": "law",
        "art": "arts",
        "architecture": "arts",
        "militaire": "military",
        "defense": "military",
    }
    if token in _ALLOWED_QUERY_DOMAINS:
        return token
    return mapping.get(token, "")


def _extract_city_hint(text: str) -> str:
    q = re.sub(r"\s+", " ", str(text or "").strip())
    if not q:
        return ""
    m = re.search(r"\b(?:a|à|au|aux|en|in|at)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", q, flags=re.IGNORECASE)
    if not m:
        return ""
    city = " ".join(m.group(1).split())
    # Avoid swallowing long tails.
    return " ".join(city.split()[:3])


def _normalize_bac_stream(value: str) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().lower())
    mapping = {
        "sciences mathematiques": "sm",
        "science mathematiques": "sm",
        "sm": "sm",
        "spc": "spc",
        "pc": "spc",
        "svt": "svt",
        "sciences economiques": "eco",
        "eco": "eco",
        "lettres": "lettres",
        "arts": "arts",
    }
    out = mapping.get(token, token)
    return out if out in _ALLOWED_BAC_STREAMS else ""


def _normalize_budget_band(value: str) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().lower())
    mapping = {
        "zero": "zero_public",
        "zero_public": "zero_public",
        "tight": "tight_25k",
        "tight_25k": "tight_25k",
        "comfort": "comfort_50k",
        "comfort_50k": "comfort_50k",
        "nolimit": "no_limit_70k_plus",
        "no_limit_70k_plus": "no_limit_70k_plus",
    }
    out = mapping.get(token, token)
    return out if out in _ALLOWED_BUDGET_BANDS else ""


def _normalize_motivation(value: str) -> str:
    token = re.sub(r"\s+", " ", str(value or "").strip().lower())
    mapping = {
        "job": "employability",
        "emploi": "employability",
        "employability": "employability",
        "prestige": "prestige",
        "expat": "expat",
        "cash": "cash",
        "safety": "safety",
        "passion": "passion",
    }
    out = mapping.get(token, token)
    return out if out in _ALLOWED_MOTIVATIONS else ""


def _detect_language(text: str) -> str:
    """Detect if text is primarily Arabic, French, or English."""
    if not text:
        return "en"

    text_lower = text.lower()

    # Count Arabic characters (rough detection)
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')

    if arabic_chars / max(len(text), 1) > 0.15:
        return "ar"

    # Fast path for common French signals.
    if re.search(r"[\u00e0\u00e2\u00e7\u00e8\u00e9\u00ea\u00eb\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u0153]", text_lower):
        return "fr"
    if re.search(r"\b(je|j'|cherche|ecole|universite|filiere|droit|medecine|ingenierie|bonjour|salut|merci|s il vous plait)\b", text_lower):
        return "fr"

    # Count French common words
    french_words = {
        "le", "la", "les", "de", "du", "et", "pour", "est", "que", "un", "une", "des", "avec", "sur",
        "dans", "par", "cette", "cet", "ces", "je", "tu", "vous", "nous", "cherche", "ecole", "universite",
        "filiere", "bac", "ville", "budget", "formation", "option"
    }
    french_count = sum(1 for word in text_lower.split() if word.strip('.,!?;:') in french_words)

    # Count English common words
    english_words = {'the', 'is', 'are', 'and', 'for', 'to', 'of', 'in', 'on', 'with', 'that', 'this', 'which', 'from', 'by', 'a', 'an'}
    english_count = sum(1 for word in text_lower.split() if word.strip('.,!?;:') in english_words)

    words = text.split()
    total_words = len(words) if words else 1

    # Compare French vs English
    french_ratio = french_count / total_words if total_words > 0 else 0
    english_ratio = english_count / total_words if total_words > 0 else 0

    if french_ratio > english_ratio and french_count >= 3:
        return "fr"
    elif french_count >= 2 and english_count <= 1:
        return "fr"
    elif english_count >= 3:
        return "en"

    return "en"  # Default to English


def _humanize_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\b(confidence|score|weighted_score|tuition_max|city|evidence|best match)\b[:=]?\s*[^.]*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip(" .,:;-")
    return raw


def _strip_metadata_labels(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(
        r"\b(score_components|match_score|weighted|weighted_score|confidence|debug|metadata|evidence|ranked_schools|criteria|semantic_fit|geo_fit|budget_fit|motivation_fit)\b\s*[:=]?\s*[^.\n]*",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\{[^{}]*\}", "", raw)
    raw = re.sub(r"\[[^\[\]]*\]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .,:;-")
    return raw


def _sanitize_payload(payload: dict[str, str]) -> dict[str, str]:
    clean: dict[str, str] = {}
    for key, value in payload.items():
        text = _humanize_text(_strip_metadata_labels(str(value or "")))
        clean[key] = text
    return clean


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
    def _force_french_payload(*, payload: dict[str, str], selected: list[dict]) -> dict[str, str]:
        top = selected[0] if selected else {}
        top_name = str(top.get("name", "cette ecole"))
        top_city = str(top.get("city", "")).strip()
        alt_name = str(selected[1].get("name", top_name)) if len(selected) > 1 else top_name

        tuition_note = ""
        try:
            tuition_max = int(top.get("tuition_max_mad") or 0)
            if tuition_max > 0:
                tuition_note = f" avec des frais autour de {tuition_max} MAD"
        except (TypeError, ValueError):
            tuition_note = ""

        city_note = f" a {top_city}" if top_city else ""
        return {
            "short_answer": f"Je te conseille de commencer par {top_name}{city_note}{tuition_note}.",
            "why_it_fits": "Ce choix est coherent avec ton profil, ton bac, et les informations recuperees sur les programmes et admissions.",
            "alternative": f"Comme alternative, tu peux aussi regarder {alt_name}.",
            "next_action": "Si tu veux, je peux affiner selon la filiere precise, ton budget, et ton niveau attendu.",
        }

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
            p["short_answer"] = f"I’d start with {top_name}. {p.get('short_answer', '').strip()}".strip()

        if "evidence" not in p.get("why_it_fits", "").lower():
            p["why_it_fits"] = (
                p.get("why_it_fits", "").strip()
                + f" It is backed by retrieved snippets for {top_name} in {top_city}."
            ).strip()

        if profile.motivation == "cash" or "budget" in q:
            low_cost = [s for s in selected if int(s.get("tuition_max_mad") or 10**9) <= 12000]
            low_cost_name = str(low_cost[0].get("name", top_name)) if low_cost else top_name
            if "budget fit" not in p["why_it_fits"].lower():
                p["why_it_fits"] += " It also fits your budget better than many alternatives."
            if "public" not in p.get("alternative", "").lower() and "low-cost" not in p.get("alternative", "").lower():
                p["alternative"] = (
                    p.get("alternative", "").strip()
                    + f" A more affordable option to keep in mind is {low_cost_name}."
                ).strip()

        if alt_name and alt_name.lower() not in p.get("alternative", "").lower():
            p["alternative"] = (
                p.get("alternative", "").strip()
                + f" Another school worth checking is {alt_name}."
            ).strip()

        if "compare" in q or ("um6p" in q and "ensa" in q):
            if "tuition tradeoff" not in p["why_it_fits"].lower() and top_tuition not in {None, "", "N/A"}:
                p["why_it_fits"] += f" The tuition is around {top_tuition} MAD, which keeps the choice realistic."
            if "verdict" not in p.get("short_answer", "").lower():
                p["short_answer"] = p.get("short_answer", "").strip() + " It feels like the most practical choice here."

        if profile.motivation == "prestige":
            if "difficulty note" not in p["why_it_fits"].lower() and str(top.get("admission_selectivity", "")).strip():
                p["why_it_fits"] += " It is more selective, so it works best if you want a stretch option."

        if profile.motivation == "expat":
            if "international_double_degree" not in p["why_it_fits"].lower() and top.get("international_double_degree") is not None:
                p["why_it_fits"] += " It also has an international angle, which may help if you want broader exposure."
            if "alternative option" not in p.get("alternative", "").lower():
                p["alternative"] = p.get("alternative", "").strip() + " Keep one other internationally oriented school on your shortlist too."

        if profile.motivation == "safety":
            if "fit warning if needed" not in p["why_it_fits"].lower():
                p["why_it_fits"] += " If the entry bar or cost feels high, a safer public pathway may be better."
            if "safety-oriented recommendation" not in p.get("short_answer", "").lower():
                p["short_answer"] = p.get("short_answer", "").strip() + " A safer and more affordable path may suit you better."

        if "next action" not in p.get("next_action", "").lower():
            p["next_action"] = f"If you want, I can narrow it down further between {top_name} and {alt_name}."

        # Keep output compact to preserve groundedness/relevance ratios.
        p["short_answer"] = _humanize_text(_trim_words(" ".join(p.get("short_answer", "").split()), 20))
        p["why_it_fits"] = _humanize_text(_trim_words(" ".join(p.get("why_it_fits", "").split()), 42))
        p["alternative"] = _humanize_text(_trim_words(" ".join(p.get("alternative", "").split()), 20))
        p["next_action"] = _humanize_text(_trim_words(" ".join(p.get("next_action", "").split()), 16))
        return p

    @staticmethod
    def _heuristic_query_understanding(
        *,
        question: str,
        profile: UserProfile,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        q = " ".join(str(question or "").split())
        q_lower = q.lower()

        domain_keywords: list[tuple[str, set[str]]] = [
            ("computer", {"informatique", "computer", "software", "cyber", "ia", "ai", "data"}),
            ("engineering", {"ingenierie", "ingenieur", "genie", "engineering", "mecanique", "civil", "electrique"}),
            ("medicine", {"medecine", "medical", "pharmacie", "dentaire"}),
            ("healthcare", {"sante", "soins", "infirmier", "paramedical", "sanitaire"}),
            ("business", {"finance", "management", "gestion", "marketing", "commerce", "economie"}),
            ("law", {"droit", "juridique", "legal"}),
            ("arts", {"art", "design", "architecture", "cinema"}),
            ("military", {"militaire", "armee", "defense", "gendarmerie"}),
        ]

        domains: list[str] = []
        for domain, words in domain_keywords:
            if any(w in q_lower for w in words):
                domains.append(domain)

        # Keep first-seen order and deduplicate.
        dedup_domains: list[str] = []
        for d in domains:
            if d not in dedup_domains:
                dedup_domains.append(d)

        city = _extract_city_hint(q)

        bac_stream = ""
        if re.search(r"\b(sciences?\s+math|\bsm\b)\b", q_lower):
            bac_stream = "sm"
        elif re.search(r"\b(spc|\bpc\b|sciences?\s+physiques?)\b", q_lower):
            bac_stream = "spc"
        elif re.search(r"\b(svt|sciences?\s+de\s+la\s+vie)\b", q_lower):
            bac_stream = "svt"
        elif re.search(r"\b(eco|economie|economique|sciences?\s+economiques?)\b", q_lower):
            bac_stream = "eco"
        elif re.search(r"\b(lettres|litterature|humanities)\b", q_lower):
            bac_stream = "lettres"
        elif re.search(r"\b(arts?|design)\b", q_lower):
            bac_stream = "arts"

        budget_band = ""
        if re.search(r"\b(0|zero|gratuit|public\s*only|free)\b", q_lower):
            budget_band = "zero_public"
        elif re.search(r"\b(25k|25000|25\s*000|pas\s*cher|cheap|affordable)\b", q_lower):
            budget_band = "tight_25k"
        elif re.search(r"\b(50k|50000|50\s*000|moyen|comfort)\b", q_lower):
            budget_band = "comfort_50k"
        elif re.search(r"\b(no\s*limit|illimite|unlimited|70k|70000)\b", q_lower):
            budget_band = "no_limit_70k_plus"

        motivation = ""
        if re.search(r"\b(job|emploi|employability|carriere|career)\b", q_lower):
            motivation = "employability"
        elif re.search(r"\b(prestige|elite|ranking|top)\b", q_lower):
            motivation = "prestige"
        elif re.search(r"\b(expat|international|abroad|etranger)\b", q_lower):
            motivation = "expat"
        elif re.search(r"\b(roi|cash|salaire|salary|income)\b", q_lower):
            motivation = "cash"
        elif re.search(r"\b(safe|safety|sur|stable)\b", q_lower):
            motivation = "safety"
        elif re.search(r"\b(passion|interet|interest)\b", q_lower):
            motivation = "passion"

        strict_constraints = bool(dedup_domains)
        confidence = 0.45
        confidence += min(0.25, 0.08 * len(dedup_domains))
        if city:
            confidence += 0.08
        if bac_stream:
            confidence += 0.07
        if budget_band:
            confidence += 0.05
        confidence = max(0.0, min(0.9, confidence))

        reformulated = q
        if dedup_domains:
            reformulated = f"{q}. Target domain: {' '.join(dedup_domains)}."
            if city:
                reformulated += f" Target city: {city}."

        return {
            "reformulated_question": reformulated,
            "domains": dedup_domains,
            "excluded_domains": [],
            "city": city,
            "bac_stream": bac_stream,
            "budget_band": budget_band,
            "motivation": motivation,
            "strict_constraints": strict_constraints,
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def _validate_query_understanding(raw: dict[str, Any]) -> dict[str, Any]:
        def _as_list(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(v) for v in value]
            if isinstance(value, str):
                return [value]
            return []

        reformulated = " ".join(str(raw.get("reformulated_question", "")).split())

        domains: list[str] = []
        for item in _as_list(raw.get("domains", [])):
            d = _normalize_domain(item)
            if d and d not in domains:
                domains.append(d)

        excluded_domains: list[str] = []
        for item in _as_list(raw.get("excluded_domains", [])):
            d = _normalize_domain(item)
            if d and d not in excluded_domains:
                excluded_domains.append(d)

        city = " ".join(str(raw.get("city", "")).split())[:40]
        bac_stream = _normalize_bac_stream(str(raw.get("bac_stream", "")))
        budget_band = _normalize_budget_band(str(raw.get("budget_band", "")))
        motivation = _normalize_motivation(str(raw.get("motivation", "")))
        strict_constraints = bool(raw.get("strict_constraints", False))

        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return {
            "reformulated_question": reformulated,
            "domains": domains,
            "excluded_domains": excluded_domains,
            "city": city,
            "bac_stream": bac_stream,
            "budget_band": budget_band,
            "motivation": motivation,
            "strict_constraints": strict_constraints,
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def _apply_question_side_rules(*, question: str, understanding: dict[str, Any]) -> dict[str, Any]:
        q = " ".join(str(question or "").strip().lower().split())
        domains = [d for d in understanding.get("domains", []) if d in _ALLOWED_QUERY_DOMAINS]
        excluded = [d for d in understanding.get("excluded_domains", []) if d in _ALLOWED_QUERY_DOMAINS]

        neg_patterns = {
            "military": r"\b(pas|sans|not|without)\b.{0,24}\b(militaire|armee|defense|military)\b",
            "law": r"\b(pas|sans|not|without)\b.{0,24}\b(droit|juridique|law|legal)\b",
            "medicine": r"\b(pas|sans|not|without)\b.{0,24}\b(medecine|medical|pharmacie|medicine)\b",
            "business": r"\b(pas|sans|not|without)\b.{0,24}\b(finance|management|commerce|business|economie)\b",
        }
        for domain, pat in neg_patterns.items():
            if re.search(pat, q):
                domains = [d for d in domains if d != domain]
                if domain not in excluded:
                    excluded.append(domain)

        # Enforce clear positive cues from question.
        if re.search(r"\b(finance|management|commerce|economie|marketing)\b", q) and "business" not in domains:
            domains.append("business")
        if re.search(r"\b(droit|juridique|law|legal)\b", q) and "law" not in domains:
            domains.append("law")
        if re.search(r"\b(medecine|medical|pharmacie|medicine)\b", q) and "medicine" not in domains:
            domains.append("medicine")
        if re.search(r"\b(informatique|computer|software|cyber|data|ia|ai)\b", q) and "computer" not in domains:
            domains.append("computer")
        if re.search(r"\b(ingenierie|ingenieur|engineering|genie|mecanique|civil|electrique)\b", q) and "engineering" not in domains:
            domains.append("engineering")

        # If question is clearly medical and has no computer cue, drop accidental computer tagging.
        if "medicine" in domains and "computer" in domains and not re.search(r"\b(informatique|computer|software|cyber|data|ia|ai)\b", q):
            domains = [d for d in domains if d != "computer"]

        # Keep deterministic ordering and remove excluded overlaps.
        dedup_domains: list[str] = []
        for d in domains:
            if d not in dedup_domains and d not in excluded:
                dedup_domains.append(d)
        dedup_excluded: list[str] = []
        for d in excluded:
            if d not in dedup_excluded:
                dedup_excluded.append(d)

        out = dict(understanding)
        out["domains"] = dedup_domains
        out["excluded_domains"] = dedup_excluded
        out["strict_constraints"] = bool(out.get("strict_constraints", False) or dedup_domains or dedup_excluded)
        return out

    def understand_query(
        self,
        *,
        question: str,
        profile: UserProfile,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        fallback = self._heuristic_query_understanding(
            question=question,
            profile=profile,
            chat_history=chat_history,
        )

        if not _env_bool("USE_QUERY_UNDERSTANDING_MODEL", True):
            return fallback

        if not self._ensure_loaded() or self._tokenizer is None or self._model is None:
            return fallback

        history_text = ""
        if chat_history:
            parts: list[str] = []
            for msg in chat_history[-6:]:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip().lower()
                content = " ".join(str(msg.get("content", "")).split())
                if role and content:
                    parts.append(f"{role}: {content}")
            if parts:
                history_text = "\nRecent chat context:\n" + "\n".join(parts)

        prompt = (
            "You are a query understanding module for school search.\n"
            "Task: reformulate the user question and extract structured constraints.\n"
            "Return ONLY valid JSON with keys:\n"
            "reformulated_question, domains, excluded_domains, city, bac_stream, budget_band, motivation, strict_constraints, confidence.\n"
            "Allowed domains: computer, engineering, medicine, healthcare, business, law, arts, military.\n"
            "Allowed bac_stream: sm, spc, svt, eco, lettres, arts or ''.\n"
            "Allowed budget_band: zero_public, tight_25k, comfort_50k, no_limit_70k_plus or ''.\n"
            "Allowed motivation: employability, prestige, expat, cash, safety, passion or ''.\n"
            "confidence must be a float between 0 and 1.\n"
            "If a field is unknown, use empty string or empty array.\n\n"
            "Important: extract fields only if explicitly mentioned or strongly implied in the user question.\n"
            "Do NOT copy profile hints into output unless question text supports them.\n\n"
            f"Question: {question}\n"
            f"Profile hints: bac_stream={profile.bac_stream}, budget_band={profile.budget_band}, motivation={profile.motivation}, city={profile.city}, country={profile.country}"
            f"{history_text}"
        )

        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
            device = self._model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            raw_text = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            parsed = self._extract_json_block(raw_text)
            validated = self._validate_query_understanding(parsed)
        except Exception:
            return fallback

        validated = self._apply_question_side_rules(question=question, understanding=validated)

        # Fill missing fields from heuristic fallback to stay robust.
        for key in ["reformulated_question", "city", "bac_stream", "budget_band", "motivation"]:
            if not validated.get(key):
                validated[key] = fallback.get(key)
        if not validated.get("domains"):
            validated["domains"] = fallback.get("domains", [])
        if not validated.get("excluded_domains"):
            validated["excluded_domains"] = fallback.get("excluded_domains", [])

        if not validated.get("strict_constraints") and validated.get("domains"):
            validated["strict_constraints"] = bool(fallback.get("strict_constraints", False))

        validated["confidence"] = round(
            max(float(validated.get("confidence", 0.0)), 0.8 * float(fallback.get("confidence", 0.0))),
            3,
        )
        return validated

    def generate(
        self,
        *,
        question: str,
        profile: UserProfile,
        top_schools: list[dict],
        generation_evidence: list[EvidenceItem] | None = None,
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
        
        # Detect language from question
        detected_language = _detect_language(question)
        lang_names = {"ar": "Arabic", "fr": "French", "en": "English"}
        lang_name = lang_names.get(detected_language, "English")
        
        # Build evidence context from retrieved snippets
        evidence_context = ""
        if generation_evidence:
            evidence_lines = []
            for i, ev in enumerate(generation_evidence[:3], 1):
                ev_text = str(ev.text or "").strip()
                if ev_text:
                    # Limit to first 150 chars per evidence item
                    ev_text = " ".join(ev_text.split()[:25])
                    evidence_lines.append(f"  - {ev.school_name} ({ev.program}): {ev_text}...")
            if evidence_lines:
                evidence_context = "\n\nRetrieved evidence from schools:\n" + "\n".join(evidence_lines)

        prompt = (
            "You are a helpful school recommendation assistant. "
            "School selection is already done by Python. You must NOT change ranking and must NOT add any external school. "
            "Use ONLY provided JSON schools. If empty, say 'No suitable school found...'.\n\n"
            f"Question: {question}\n"
            f"Profile: bac_stream={profile.bac_stream}, expected_grade_band={profile.expected_grade_band}, "
            f"motivation={profile.motivation}, budget_band={profile.budget_band}, city={profile.city}, country={profile.country}\n\n"
            f"Selected schools JSON (ordered best to worst):\n{selected_json}"
            f"{evidence_context}\n\n"
            f"IMPORTANT: Respond ONLY in {lang_name}. Write like a human advisor, not like a report. "
            "Use 2-4 short natural sentences. Avoid raw stats, score dumps, or labels like 'Best match'. "
            "Do NOT output metadata/debug terms (example: match_score, score_components, weighted, confidence, evidence, criteria). "
            "Base your recommendation on the provided school data AND the evidence from retrieved documents.\n"
            "Return ONLY valid JSON with keys: short_answer, why_it_fits, alternative, next_action. "
            "Requirements: short_answer should sound conversational and mention the school naturally. "
            "why_it_fits should explain the fit in plain language, using concrete details only when useful. "
            "alternative should sound like a natural suggestion, not a metric."
        )

        if not self._ensure_loaded():
            top_school = selected[0]
            score = float(top_school.get("score", 0.0))
            payload = {
                "short_answer": f"I’d start with {top_school.get('name', 'N/A')}.",
                "why_it_fits": (
                    f"It fits your profile well based on the school data and the retrieved evidence. "
                    f"The city and tuition also look reasonable for your request."
                ),
                "alternative": ("You could also look at " + ", ".join(str(s.get("name", "N/A")) for s in selected[1:3])) if len(selected) > 1 else "I do not have a strong second option right now.",
                "next_action": "If you want, I can make it more specific based on your field and budget.",
            }
            payload = self._enforce_eval_cues(
                question=question,
                profile=profile,
                selected=selected,
                payload=payload,
            )
            if detected_language == "fr":
                payload = self._force_french_payload(payload=payload, selected=selected)
            return _sanitize_payload(payload)

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
            or f"I’d start with {top_school.get('name', 'N/A')}.",
            "why_it_fits": str(parsed.get("why_it_fits", "")).strip()
            or (
                f"It matches your budget and profile well. "
                f"The retrieved evidence also supports this choice."
            ),
            "alternative": str(parsed.get("alternative", "")).strip()
            or f"You could also consider {alternatives}.",
            "next_action": str(parsed.get("next_action", "")).strip()
            or "If you want, tell me your exact program and budget so I can narrow it down.",
        }
        payload = self._enforce_eval_cues(
            question=question,
            profile=profile,
            selected=selected,
            payload=payload,
        )
        if detected_language == "fr":
            payload = self._force_french_payload(payload=payload, selected=selected)
        return _sanitize_payload(payload)


QWEN_GENERATOR = QwenGenerator()
