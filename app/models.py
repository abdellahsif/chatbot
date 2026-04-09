from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


def _pick_first(data: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _normalize_bac_stream(value: str) -> str:
    raw = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(raw.split())
    if not text:
        return ""

    mapping = {
        "science mathematiques": "sm",
        "sciences mathematiques": "sm",
        "science math": "sm",
        "sciences math": "sm",
        "sm": "sm",
        "pc": "spc",
        "spc": "spc",
        "physique": "spc",
        "sciences physiques": "spc",
        "svt": "svt",
        "sciences de la vie": "svt",
        "sciences de la vie et de la terre": "svt",
        "science de la vie": "svt",
        "eco": "eco",
        "economique": "eco",
        "economie": "eco",
        "sciences economiques": "eco",
        "sciences de gestion": "eco",
        "lettres": "lettres",
        "litterature": "lettres",
        "sciences humaines": "lettres",
        "arts": "arts",
        "art": "arts",
        "design": "arts",
    }
    return mapping.get(text, text)


def _normalize_grade_band(value: str) -> str:
    raw = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(raw.split())
    if not text:
        return ""

    if text in {"non renseignee", "non renseignée", "n/a", "na", "none", ""}:
        return ""

    direct_map = {
        "passable": "passable",
        "bien": "bien",
        "bein": "bien",
        "tres bien": "tres_bien",
        "très bien": "tres_bien",
        "elite": "elite",
        "excellent": "elite",
    }
    mapped = direct_map.get(text)
    if mapped:
        return mapped

    nums = [int(n) for n in re.findall(r"\d+", text)]
    if nums:
        top = max(nums)
        if top >= 18:
            return "elite"
        if top >= 16:
            return "tres_bien"
        if top >= 14:
            return "bien"
        if top >= 10:
            return "passable"

    if any(token in text for token in ["elite", "excellent", "18", "19", "20"]):
        return "elite"
    if any(token in text for token in ["tres bien", "très bien", "16", "17"]):
        return "tres_bien"
    if any(token in text for token in ["bien", "14", "15"]):
        return "bien"
    if any(token in text for token in ["passable", "10", "11", "12"]):
        return "passable"
    return text


def _normalize_country(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return "MA"
    if text in {"ma", "morocco", "maroc", "marocain", "المغرب"}:
        return "MA"
    return text.upper()


def _normalize_motivation(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if not text:
        return ""

    mapping = {
        "cash": "cash",
        "roi": "cash",
        "salary": "cash",
        "income": "cash",
        "prestige": "prestige",
        "passion": "passion",
        "interest": "passion",
        "interet": "passion",
        "safety": "safety",
        "safe": "safety",
        "stability": "safety",
        "stable": "safety",
        "expat": "expat",
        "international": "expat",
        "abroad": "expat",
        "global": "expat",
        "employability": "employability",
        "career": "employability",
        "emploi": "employability",
        "job": "employability",
        "work": "employability",
        "carriere": "employability",
    }
    return mapping.get(text, text)


def _normalize_budget_band(value: str) -> str:
    text = " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    if not text:
        return ""

    if text in {"zero public", "zero_public", "0dh", "0", "free", "gratuit", "public"}:
        return "zero_public"
    if text in {"tight 25k", "tight_25k", "serre", "serré", "low budget", "budget serre"}:
        return "tight_25k"
    if text in {"comfort 50k", "comfort_50k", "confort", "50k", "medium budget"}:
        return "comfort_50k"
    if text in {"no limit 70k plus", "no_limit_70k_plus", "illimite", "illimité", "no limit", "unlimited", "70k plus"}:
        return "no_limit_70k_plus"

    if "25k" in text:
        return "tight_25k"
    if "50k" in text:
        return "comfort_50k"
    if "70k" in text or "no limit" in text or "illim" in text:
        return "no_limit_70k_plus"

    return text


@dataclass
class UserProfile:
    bac_stream: str
    expected_grade_band: str
    motivation: str
    budget_band: str
    city: str
    country: str
    classe: str = ""
    note_esperee: str = ""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "UserProfile":
        country = _normalize_country(_pick_first(data, "country", "pays", default="Maroc"))
        city = _pick_first(data, "city", "ville")
        bac_stream = _normalize_bac_stream(_pick_first(data, "bac_stream", "serie_bac", "série_bac"))
        note_esperee = _pick_first(data, "note_esperee", "expected_grade_band")
        expected_grade_band = _normalize_grade_band(note_esperee)
        if not expected_grade_band:
            expected_grade_band = _normalize_grade_band(_pick_first(data, "expected_grade_band"))
        return UserProfile(
            bac_stream=bac_stream,
            expected_grade_band=expected_grade_band,
            motivation=_normalize_motivation(_pick_first(data, "motivation")),
            budget_band=_normalize_budget_band(_pick_first(data, "budget_band", "budget")),
            city=city,
            country=country,
            classe=_pick_first(data, "classe", "class", "niveau"),
            note_esperee=note_esperee,
        )


@dataclass
class QueryRequest:
    question: str
    profile: UserProfile
    top_k: int = 5
    chat_history: list[dict[str, str]] | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "QueryRequest":
        top_k = int(data.get("top_k", 5))
        top_k = max(1, min(10, top_k))
        profile_payload = data.get("profile", {})
        if not isinstance(profile_payload, dict):
            profile_payload = {}
        if not profile_payload:
            profile_payload = data
        return QueryRequest(
            question=str(data.get("question", "")).strip(),
            profile=UserProfile.from_dict(profile_payload),
            top_k=top_k,
            chat_history=[
                {
                    "role": str(item.get("role", "")).strip().lower(),
                    "content": str(item.get("content", "")).strip(),
                }
                for item in (data.get("chat_history", []) if isinstance(data.get("chat_history", []), list) else [])
                if isinstance(item, dict)
            ],
        )


@dataclass
class EvidenceItem:
    chunk_id: str
    school_id: str
    school_name: str
    program: str
    recorded_at: str
    text: str
    score: float


@dataclass
class QueryResponse:
    short_answer: str
    why_it_fits: str
    evidence: list[EvidenceItem]
    alternative: str
    next_action: str
    confidence: float
    message_paragraph: str = ""
    ranked_schools: list[dict[str, Any]] | None = None

    def _compose_message_paragraph(self) -> str:
        parts = [
            str(self.short_answer or "").strip(),
            str(self.why_it_fits or "").strip(),
            str(self.alternative or "").strip(),
            str(self.next_action or "").strip(),
        ]
        paragraph = " ".join(p for p in parts if p)
        return " ".join(paragraph.split())

    def to_dict(self) -> dict[str, Any]:
        message_paragraph = str(self.message_paragraph or "").strip() or self._compose_message_paragraph()
        return {
            "short_answer": self.short_answer,
            "why_it_fits": self.why_it_fits,
            "evidence": [asdict(item) for item in self.evidence],
            "alternative": self.alternative,
            "next_action": self.next_action,
            "confidence": self.confidence,
            "message_paragraph": message_paragraph,
            "ranked_schools": self.ranked_schools or [],
        }


@dataclass
class EvalResult:
    id: str
    passed: bool
    checks: dict[str, bool]
    answer_preview: str


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int
    results: list[EvalResult]
    metrics: dict[str, Any] | None = None
    log_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "results": [asdict(item) for item in self.results],
            "metrics": self.metrics or {},
            "log_path": self.log_path,
        }
