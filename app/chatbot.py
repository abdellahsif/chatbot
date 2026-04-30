from __future__ import annotations

import json
import random
import re
from collections import deque
from statistics import mean
from typing import Any

from app.generator import QWEN_GENERATOR
from app.models import EvidenceItem, QueryResponse, UserProfile
from app.retriever import resolve_effective_profile, retrieve
from app.supabase_store import fetch_user_career_profile


_BUDGET_MAX = {
    "zero_public": 0,
    "tight_25k": 25000,
    "comfort_50k": 50000,
    "no_limit_70k_plus": 10**9,
}

_BAC_LABEL = {
    "sm": "sciences math",
    "spc": "sciences physiques",
    "svt": "sciences de la vie",
    "eco": "sciences economiques",
    "lettres": "lettres",
    "arts": "arts",
}

_BUDGET_LABEL = {
    "zero_public": "public low-cost",
    "tight_25k": "up to 25k MAD",
    "comfort_50k": "up to 50k MAD",
    "no_limit_70k_plus": "70k+ MAD",
}

_MOTIVATION_LABEL = {
    "cash": "return on investment",
    "prestige": "prestige",
    "expat": "international exposure",
    "employability": "employability",
    "safety": "safe realistic path",
    "passion": "interest fit",
}

_RECENT_OUTPUTS: deque[str] = deque(maxlen=5)
_RECENT_OPENERS: deque[str] = deque(maxlen=3)
_RESPONSE_STYLES = [
    "advisor_exploratory",
    "comparison_balanced",
    "confused_guidance",
    "direct_recommendation",
    "clarification_mode",
]


def detect_language(text: str) -> str:
    q = " ".join(str(text or "").split()).strip().lower()
    if re.search(r"\b(le|la|les|un|une|des|est|pour|avec|quelle|quelles|ecole|ecoles|universite|bonjour|salut|merci)\b", q):
        return "fr"
    return "en"


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _profile_has_signal(profile: UserProfile) -> bool:
    return any(
        [
            bool((profile.bac_stream or "").strip()),
            bool((profile.expected_grade_band or "").strip()),
            bool((profile.motivation or "").strip()),
            bool((profile.budget_band or "").strip()),
            bool((profile.city or "").strip()),
        ]
    )


def _profile_to_query(profile: UserProfile) -> str:
    parts: list[str] = ["best matching schools"]

    if (profile.country or "").strip():
        parts.append(f"country {profile.country.strip()}")
    if (profile.city or "").strip():
        parts.append(f"city {profile.city.strip()}")

    bac_key = (profile.bac_stream or "").strip().lower()
    if bac_key:
        parts.append(f"bac {_BAC_LABEL.get(bac_key, bac_key)}")

    budget_key = (profile.budget_band or "").strip().lower()
    if budget_key:
        parts.append(f"budget {_BUDGET_LABEL.get(budget_key, budget_key)}")

    motivation_key = (profile.motivation or "").strip().lower()
    if motivation_key:
        parts.append(f"goal {_MOTIVATION_LABEL.get(motivation_key, motivation_key)}")

    if (profile.expected_grade_band or "").strip():
        parts.append(f"grade {profile.expected_grade_band.strip()}")

    return ". ".join(parts)


def _is_placeholder_profile_request(question: str) -> bool:
    q = " ".join(str(question or "").strip().lower().split())
    if not q:
        return False
    if q == "profile request":
        return True
    if q.startswith("profile request ("):
        return True
    return q in {
        "recommend based on profile",
        "recommendation based on profile",
        "profile recommendation",
        "recommend me based on profile",
    }


def _norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match_city(city_a: str, city_b: str) -> bool:
    a = " ".join((city_a or "").strip().lower().split())
    b = " ".join((city_b or "").strip().lower().split())
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _school_status_text(school: dict[str, Any]) -> str:
    status = str(school.get("legal_status", "")).strip()
    if status:
        return status
    typ = str(school.get("type", "")).strip().lower()
    return "public" if "public" in typ else ("private" if typ else "")


def _school_tuition_range(school: dict[str, Any]) -> tuple[int, int]:
    mn = _to_int(school.get("pricing_min"), default=-1)
    mx = _to_int(school.get("pricing_max"), default=-1)
    if mn < 0:
        mn = _to_int(school.get("tuition_min_mad"), default=0)
    if mx < 0:
        mx = _to_int(school.get("tuition_max_mad"), default=mn)
    if mx < mn:
        mx = mn
    return mn, mx


def _school_program_labels(school: dict[str, Any]) -> str:
    raw = " | ".join(
        [
            str(school.get("programs_tags", "")).strip(),
            str(school.get("filieres", "")).strip(),
        ]
    ).strip(" |")
    if raw:
        return raw
    programs = school.get("programs", [])
    if isinstance(programs, list):
        return " | ".join(str(p) for p in programs if str(p).strip())
    return ""


def _humanize_programs(text: str, max_items: int = 4) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    labels: list[str] = []
    for p in parts:
        human = " ".join(p.replace("_", " ").split())
        if human and human not in labels:
            labels.append(human)
    return ", ".join(labels[:max_items])


def _clean_program_list(school: dict[str, Any], max_items: int = 4) -> list[str]:
    raw = _school_program_labels(school)
    if not raw:
        return []
    parts = [p.strip() for p in raw.split("|") if p.strip()]
    out: list[str] = []
    for p in parts:
        item = " ".join(str(p).replace("_", " ").split())
        if item and item not in out:
            out.append(item)
        if len(out) >= max_items:
            break
    return out


def _program_tokens_from_school(school: dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            str(school.get("programs_tags", "")),
            str(school.get("filieres", "")),
            " ".join(str(p) for p in school.get("programs", []) if str(p).strip()),
        ]
    )
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    stop = {
        "de", "la", "le", "et", "des", "du", "the", "and", "of",
        "program", "programs", "programme", "programmes",
    }
    return {tok for tok in tokens if len(tok) >= 4 and tok not in stop}


def _schools_share_direction(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_tokens = _program_tokens_from_school(a)
    b_tokens = _program_tokens_from_school(b)
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens & b_tokens
    return len(overlap) >= 2 or (len(overlap) >= 1 and min(len(a_tokens), len(b_tokens)) <= 3)


def _message_signals_rejection(text: str) -> bool:
    q = " ".join(str(text or "").strip().lower().split())
    if not q:
        return False
    patterns = [
        r"\b(i do not like|i don't like|dont like|do not want|don't want|not interested|another option|something else|change direction)\b",
        r"\b(je n aime pas|j aime pas|je ne veux pas|pas interesse|pas intéressé|autre chose|une autre option|changer de voie)\b",
        r"\b(ma bghitch|mabghitch|ma3jbnich|la ma bghitch|bghit haja okhra)\b",
    ]
    return any(re.search(p, q, flags=re.IGNORECASE) for p in patterns)


def _extract_rejected_school_from_history(
    *,
    chat_history: list[dict[str, str]] | None,
    candidate_schools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not chat_history or not candidate_schools:
        return None

    last_user = ""
    last_assistant = ""
    for msg in reversed(chat_history):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = " ".join(str(msg.get("content", "")).split()).strip()
        if role == "user" and not last_user and content:
            last_user = content
        elif role == "assistant" and not last_assistant and content:
            last_assistant = content
        if last_user and last_assistant:
            break

    if not _message_signals_rejection(last_user):
        return None

    lowered = last_assistant.lower()
    for school in candidate_schools:
        name = str(school.get("name", "")).strip()
        if name and name.lower() in lowered:
            return school
    return None


def _clean_admission_text(value: Any) -> str:
    raw = " ".join(str(value or "").split()).strip().lower()
    if not raw:
        return ""
    if raw in {"medium", "high", "low", "moyen", "moyenne", "faible", "not specified", "unknown"}:
        return ""
    return " ".join(str(value or "").split()).strip()


def build_school_facts(school: dict[str, Any]) -> dict[str, Any]:
    name = str(school.get("name", "")).strip() or "this school"
    city = str(school.get("city", "")).strip()
    programs = _clean_program_list(school)
    tuition_min, tuition_max = _school_tuition_range(school)
    if tuition_min > 0 and tuition_max > 0 and tuition_min != tuition_max:
        tuition_range = f"{tuition_min}-{tuition_max} MAD"
    elif tuition_max > 0:
        tuition_range = f"{tuition_max} MAD"
    elif tuition_min > 0:
        tuition_range = f"{tuition_min} MAD"
    else:
        tuition_range = "not specified"
    admission_requirements = (
        _clean_admission_text(school.get("admission_requirements", ""))
        or _clean_admission_text(school.get("admission_policy", ""))
        or _clean_admission_text(school.get("admission_selectivity", ""))
        or "not specified"
    )
    return {
        "name": name,
        "city": city,
        "programs": programs,
        "tuition_range": tuition_range,
        "admission_requirements": admission_requirements,
    }


def _infer_dialogue_mode(question: str) -> str:
    q = " ".join(str(question or "").lower().split())
    if not q:
        return "explore"
    if re.search(r"\b(compare|comparison|versus|vs|diff|difference|ENSIAS\s+and\s+ENSA|ensias|ensa)\b", q, flags=re.IGNORECASE):
        return "compare"
    if re.search(r"\b(i don't know|dont know|not sure|confused|lost|help me choose|what should i choose|what should i pick|ma3rftch|mab9itch 3arf)\b", q, flags=re.IGNORECASE):
        return "confused"
    if re.search(r"\b(best|top|recommend|recommendation|which school|suggest)\b", q, flags=re.IGNORECASE):
        return "direct"
    if re.search(r"\b(explore|options|schools?)\b", q, flags=re.IGNORECASE):
        return "explore"
    if len(q.split()) <= 4:
        return "vague"
    return "direct"


def _intent_from_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m == "compare":
        return "compare"
    if m == "confused":
        return "confused"
    if m in {"explore", "vague"}:
        return "explore"
    return "direct"


def _opening_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(str(text or "").split()).strip(), maxsplit=1)
    return parts[0].strip().lower() if parts and parts[0].strip() else ""


def _recent_openings(last_n: int = 3) -> set[str]:
    openings: set[str] = set()
    recent = list(_RECENT_OUTPUTS)[-last_n:]
    for item in recent:
        op = _opening_sentence(item)
        if op:
            openings.add(op)
    return openings


def _record_output(text: str) -> None:
    raw = " ".join(str(text or "").split()).strip()
    if raw:
        _RECENT_OUTPUTS.append(raw)


def _record_opener(opener: str) -> None:
    clean = " ".join(str(opener or "").split()).strip()
    if clean:
        _RECENT_OPENERS.append(clean)


def _looks_structured_output(text: str) -> bool:
    raw = " ".join(str(text or "").split()).strip().lower()
    if not raw:
        return True
    if re.search(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", str(text or "")):
        return True
    if re.search(r"\b(programs?|tuition|admission|status|score|criteria|rank)\s*[:=]", raw):
        return True
    return False


def _select_style(*, mode: str, confidence: float, top_gap: float, has_school: bool) -> str:
    if not has_school:
        return "clarification_mode"
    if mode == "compare":
        return "comparison_balanced"
    if mode == "confused":
        return "confused_guidance"
    if mode in {"vague", "explore"}:
        return "advisor_exploratory"
    return "direct_recommendation"


def _alternate_style(style: str) -> str:
    order = {
        "clarification_mode": "clarification_mode",
        "advisor_exploratory": "advisor_exploratory",
        "comparison_balanced": "comparison_balanced",
        "confused_guidance": "confused_guidance",
        "direct_recommendation": "direct_recommendation",
    }
    return order.get(style, "advisor_exploratory")


def _style_openings(style: str) -> list[str]:
    if style == "comparison_balanced":
        return [
            "If you put these side by side, the real difference shows up in how each one fits your priorities",
            "The choice here really comes down to how you weigh cost, selectivity, and learning environment",
            "A fair comparison here is about tradeoffs, not labels",
        ]
    if style == "confused_guidance":
        return [
            "We can narrow this down step by step",
            "Let us make this easier by focusing on one thing first",
            "You do not need to decide everything at once",
        ]
    if style == "clarification_mode":
        return [
            "Before I suggest a school, I need one detail from you",
            "I can guide you better with one quick clarification",
            "To avoid giving a random recommendation, I need one preference first",
        ]
    if style == "advisor_exploratory":
        return [
            "There are a couple of directions you can take here",
            "You are not locked into one path yet, which is a good thing",
            "You can keep your options open while we narrow what matters most",
        ]
    return [
        "Based on what you are looking for, one option stands out",
        "There is a direction here that could make sense for you",
        "From your profile, one path looks especially aligned",
    ]


def _pick_opening(style: str) -> str:
    used = {item.lower() for item in _RECENT_OPENERS}
    options = _style_openings(style)
    choices = [opener for opener in options if opener.lower() not in used]
    picked = random.choice(choices or options)
    _record_opener(picked)
    return picked


def _extract_detected_city(question: str, schools: list[dict[str, Any]]) -> str:
    q = " ".join(str(question or "").split()).strip().lower()
    if not q:
        return ""
    # Prefer explicit mention of known candidate cities.
    for school in schools:
        city = " ".join(str(school.get("city", "")).split()).strip()
        if city and city.lower() in q:
            return city
    m = re.search(r"\b(?:in|at|a|à|au|aux|en)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", q, flags=re.IGNORECASE)
    if not m:
        return ""
    return " ".join(m.group(1).split()[:3]).strip()


def _compose_why_from_facts(*, facts: dict[str, Any], profile: UserProfile, question: str = "") -> str:
    name = str(facts.get("name", "This school")).strip() or "This school"
    city = str(facts.get("city", "")).strip()
    programs = facts.get("programs", [])
    mode = _infer_dialogue_mode(question)

    has_tech = any(re.search(r"informatique|computer|software|data|cyber|ia|ai|engineering|ingenierie", str(p), flags=re.IGNORECASE) for p in programs)
    has_business = any(re.search(r"gestion|management|marketing|commerce|finance", str(p), flags=re.IGNORECASE) for p in programs)

    intent_hint = ""
    if mode == "confused":
        intent_hint = "You do not need to lock everything right away"
    elif mode == "compare":
        intent_hint = "What matters most here is tradeoff, not labels"
    elif mode == "vague":
        intent_hint = "To make this useful, we can keep it simple"
    else:
        intent_hint = "In your case"

    fit_hint = ""
    if has_tech and has_business:
        fit_hint = "it keeps your options open between technical and management directions"
    elif has_tech:
        fit_hint = "it stays aligned with a practical technical direction"
    elif has_business:
        fit_hint = "it leans toward a management-oriented path with concrete outcomes"
    else:
        fit_hint = "it can still be a practical base while you refine your direction"

    budget_hint = ""
    if profile.budget_band in {"zero_public", "tight_25k"}:
        budget_hint = "and it remains realistic for a tighter budget"

    where_hint = f" around {city}" if city else ""
    if mode == "compare":
        if budget_hint:
            return f"{intent_hint}; with {name}{where_hint}, you get a direction that stays grounded and {fit_hint}, {budget_hint}."
        return f"{intent_hint}; with {name}{where_hint}, you get a direction that stays grounded and {fit_hint}."
    if mode == "confused":
        if budget_hint:
            return f"{intent_hint}. {name}{where_hint} is usually easier to start with because {fit_hint}, {budget_hint}."
        return f"{intent_hint}. {name}{where_hint} is usually easier to start with because {fit_hint}."
    if budget_hint:
        return f"{intent_hint}, {name}{where_hint} can work well as a first step since {fit_hint}, {budget_hint}."
    return f"{intent_hint}, {name}{where_hint} can work well as a first step since {fit_hint}."


def _compose_alternative_from_facts(*, facts: dict[str, Any], question: str = "") -> str:
    name = str(facts.get("name", "this school")).strip() or "this school"
    city = str(facts.get("city", "")).strip()
    mode = _infer_dialogue_mode(question)
    if city:
        if mode == "compare":
            return f"To keep the comparison honest, I would also keep {name} in {city} on the table."
        if mode == "confused":
            return f"If you want a backup that still feels manageable, {name} in {city} is a reasonable second option."
        return f"Another direction you could explore is {name} in {city}."
    if mode == "compare":
        return f"To keep the comparison honest, I would also keep {name} on the table."
    if mode == "confused":
        return f"If you want a backup that still feels manageable, {name} is a reasonable second option."
    return f"Another direction you could explore is {name}."


def _select_alternative_hit(
    *,
    hits: list[dict],
    question: str,
    profile: UserProfile,
    rejected_school: dict[str, Any] | None = None,
) -> dict | None:
    if len(hits) <= 1:
        return None

    q_tokens = _norm_tokens(question)
    wants_affordable = bool(
        q_tokens
        & {
            "affordable",
            "cheap",
            "low",
            "cost",
            "budget",
            "public",
            "ofppt",
        }
    ) or profile.budget_band in {"zero_public", "tight_25k"}
    wants_quick = bool(q_tokens & {"quick", "fast", "short", "credential", "certificate", "cert"})
    wants_tech = bool(
        q_tokens
        & {
            "it",
            "tech",
            "software",
            "computer",
            "cs",
            "informatique",
            "developpement",
            "development",
            "cyber",
            "data",
        }
    )

    top_school = hits[0].get("school", {})
    top_school_id = str(top_school.get("school_id", ""))
    top_type = str(top_school.get("type", "")).lower()
    top_tokens = _norm_tokens(f"{top_school.get('name', '')} {top_type}")
    top_city = str(top_school.get("city", ""))
    top_is_public = "public" in top_type
    top_is_vocational_like = bool(top_tokens & {"ofppt", "formation", "professionnelle", "ista", "technicien", "est"})

    budget_cap = _BUDGET_MAX.get(profile.budget_band, _BUDGET_MAX["comfort_50k"])
    best_hit: dict | None = None
    best_score = float("-inf")

    for rank, hit in enumerate(hits[1:], start=1):
        school = hit.get("school", {})
        school_id = str(school.get("school_id", ""))
        if school_id and school_id == top_school_id:
            continue
        if rejected_school and _schools_share_direction(school, rejected_school):
            continue

        chunk = hit.get("chunk", {})
        name = str(school.get("name", ""))
        school_type = str(school.get("type", "")).lower()
        school_status = _school_status_text(school).lower()
        school_city = str(school.get("city", ""))
        programs = _school_program_labels(school)
        chunk_text = str(chunk.get("text", ""))
        school_tokens = _norm_tokens(f"{name} {school_type} {school_status} {programs} {chunk.get('program', '')} {chunk_text}")
        _, tuition_max = _school_tuition_range(school)
        if tuition_max <= 0:
            tuition_max = 10**9
        selectivity = str(school.get("admission_selectivity", "")).strip().lower()

        score = float(hit.get("score", 0.0))
        score += max(0.0, 0.03 - 0.005 * rank)

        if wants_tech:
            score += 0.08 if (school_tokens & {"it", "tech", "software", "computer", "informatique", "developpement", "cyber", "data"}) else -0.08

        if wants_affordable:
            if tuition_max <= 12000:
                score += 0.12
            elif tuition_max <= 25000:
                score += 0.08
            elif tuition_max <= max(25000, budget_cap):
                score += 0.03
            else:
                score -= 0.12

        if wants_quick:
            quick_signals = {"ofppt", "formation", "professionnelle", "technicien", "specialisation", "est", "ista"}
            if school_tokens & quick_signals:
                score += 0.07
            if selectivity == "high":
                score -= 0.08

        if top_is_public and ("public" in school_type or "public" in school_status):
            score += 0.04
        if top_is_vocational_like and school_tokens & {"ofppt", "formation", "professionnelle", "technicien", "est", "ista"}:
            score += 0.06

        if _match_city(school_city, profile.city) or _match_city(school_city, top_city):
            score += 0.03

        if wants_affordable and wants_quick and wants_tech:
            if "ensias" in school_tokens:
                score -= 0.2
            if selectivity == "high":
                score -= 0.06

        if score > best_score:
            best_score = score
            best_hit = hit

    return best_hit


def _is_greeting_or_low_intent(question: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return True

    tokens = re.findall(r"[a-z0-9']+", q)
    if not tokens:
        return True

    greeting_tokens = {
        "hi",
        "hello",
        "hey",
        "yo",
        "salut",
        "bonjour",
        "bonsoir",
        "salam",
        "slm",
        "marhba",
        "coucou",
        "thanks",
        "thank",
        "help",
        "aide",
    }
    filler_tokens = {"there", "bot", "chatbot", "please", "pls", "me"}

    if len(tokens) <= 3 and all(t in greeting_tokens or t in filler_tokens for t in tokens):
        return True
    return False


def _is_identity_question(question: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False
    return bool(
        re.search(
            r"\b(who\s+are\s+you|your\s+name|what\s*'?s\s+your\s+name|whats\s+your\s+name|who\s+r\s+u|ur\s+name)\b",
            q,
        )
    )


def _looks_french(question: str) -> bool:
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False
    if re.search(r"[\u00e0\u00e2\u00e7\u00e8\u00e9\u00ea\u00eb\u00ee\u00ef\u00f4\u00f9\u00fb\u00fc\u0153]", q):
        return True
    tokens = _norm_tokens(q)
    french_markers = {
        "je",
        "cherche",
        "ecole",
        "ecoles",
        "universite",
        "filiere",
        "bac",
        "droit",
        "medecine",
        "ingenierie",
        "bonjour",
        "salut",
    }
    return len(tokens & french_markers) >= 2


def _question_mentions_city(question: str, city: str) -> bool:
    q = " ".join(str(question or "").strip().lower().split())
    c = " ".join(str(city or "").strip().lower().split())
    return bool(q and c and c in q)


def _question_mentions_bac(question: str, bac_stream: str) -> bool:
    q = " ".join(str(question or "").strip().lower().split())
    bac = str(bac_stream or "").strip().lower()
    patterns = {
        "sm": r"\b(sciences?\s+math|\bsm\b)\b",
        "spc": r"\b(spc|\bpc\b|sciences?\s+physiques?)\b",
        "svt": r"\b(svt|sciences?\s+de\s+la\s+vie)\b",
        "eco": r"\b(eco|economique|economie|sciences?\s+economiques?)\b",
        "lettres": r"\b(lettres|litterature|humanities)\b",
        "arts": r"\b(arts?|design)\b",
    }
    pat = patterns.get(bac)
    return bool(pat and re.search(pat, q))


def _question_mentions_budget(question: str) -> bool:
    q = " ".join(str(question or "").strip().lower().split())
    return bool(re.search(r"\b(budget|cheap|affordable|pas\s*cher|25k|50k|70k|gratuit|free|illimite|no\s*limit)\b", q))


def _question_mentions_motivation(question: str) -> bool:
    q = " ".join(str(question or "").strip().lower().split())
    return bool(re.search(r"\b(employability|emploi|career|prestige|expat|international|abroad|roi|salary|safe|safety|passion)\b", q))


def _has_program_intent(question: str) -> bool:
    tokens = _norm_tokens(question)
    if not tokens:
        return False

    program_terms = {
        "it",
        "tech",
        "software",
        "computer",
        "cs",
        "informatique",
        "engineering",
        "business",
        "management",
        "architecture",
        "health",
        "medical",
        "paramedical",
        "design",
        "arts",
        "law",
        "data",
        "cyber",
    }
    return bool(tokens & program_terms)


def _is_city_only_school_request(question: str, profile: UserProfile) -> bool:
    tokens = _norm_tokens(question)
    if not tokens:
        return False

    if _has_program_intent(question):
        return False

    school_terms = {"school", "schools", "university", "universities", "ecole", "ecoles", "universite", "universites"}
    has_school_term = bool(tokens & school_terms)
    city_tokens = _norm_tokens(profile.city)
    has_city_mention = bool(city_tokens and (tokens & city_tokens))
    return has_school_term and has_city_mention


def _has_conflicting_constraints(question: str, profile: UserProfile) -> bool:
    q_tokens = _norm_tokens(question)
    low_budget = profile.budget_band in {"zero_public", "tight_25k"} or bool(
        q_tokens & {"affordable", "cheap", "low", "budget", "public"}
    )
    high_aspiration = profile.motivation in {"prestige", "expat"} or bool(
        q_tokens & {"prestige", "elite", "international", "global", "abroad", "top"}
    )
    return low_budget and high_aspiration


def _select_generation_evidence(evidence: list[EvidenceItem], max_items: int = 3) -> list[EvidenceItem]:
    if not evidence:
        return []
    selected: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.school_id.strip().lower() or item.school_name.strip().lower()
        if key in seen:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= max_items:
            break
    return selected or evidence[:1]


def _content_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
        "is",
        "are",
        "this",
        "that",
        "your",
        "you",
        "can",
        "will",
        "from",
        "based",
        "option",
    }
    return {t for t in _norm_tokens(text) if t not in stop}


def _sanitize_user_facing_text(text: str) -> str:
    raw = " ".join(str(text or "").split()).strip()
    if not raw:
        return ""

    # Recover structured output if the model leaked JSON.
    if raw.startswith("{") and any(k in raw for k in ['"short_answer"', '"why_it_fits"', '"alternative"', '"next_action"']):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parts = [
                    str(parsed.get("short_answer", "")).strip(),
                    str(parsed.get("why_it_fits", "")).strip(),
                    str(parsed.get("alternative", "")).strip(),
                    str(parsed.get("next_action", "")).strip(),
                ]
                raw = " ".join(p for p in parts if p)
        except Exception:
            pass

    raw = re.sub(
        r"\b(short_answer|why_it_fits|alternative|next_action|confidence|ranked_schools|score_components|match_score|criteria)\b\s*[:=]?",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    raw = re.sub(r"\bEvidence for\b", "For", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\bNo alternative available\.?", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[{}\[\]\"]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" ,.;:-")
    return raw


def _chat_continuity_fallback(message: str, chat_history: list[dict[str, str]] | None = None) -> str:
    user_msg = " ".join(str(message or "").strip().lower().split())
    last_assistant = ""
    if chat_history:
        for msg in reversed(chat_history):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = " ".join(str(msg.get("content", "")).split()).strip()
            if role == "assistant" and content:
                last_assistant = content
                break

    is_brief_ack = bool(
        re.fullmatch(
            r"(yes|yeah|yep|ok|okay|sure|go on|continue|right|exactly|true|"
            r"oui|daccord|d'accord|safi|wakha|no|nope|non|la)",
            user_msg,
        )
    )
    if last_assistant and is_brief_ack:
        return "Great, let us continue. Share a little more detail about your situation and I will help step by step."
    if last_assistant:
        return "I understand. Tell me a bit more and I will help you continue from there."
    return "Hi! I am here with you. Tell me what you want to talk about."


def _enforce_grounded_response(
    *,
    short_answer: str,
    why_it_fits: str,
    alternative: str,
    next_action: str,
    top_school: dict[str, Any],
    alt_school: dict[str, Any] | None,
    profile: UserProfile,
    question: str = "",
    style: str = "advisor_exploratory",
    suggest_school: bool = True,
) -> tuple[str, str, str, str]:
    top_facts = build_school_facts(top_school)
    alt_facts = build_school_facts(alt_school or top_school)
    top_name = str(top_facts.get("name", "this school")).strip() or "this school"
    alt_name = str(alt_facts.get("name", "")).strip()
    opening = _pick_opening(style)
    intent = _intent_from_mode(_infer_dialogue_mode(question))

    if not suggest_school:
        short_answer = f"{opening}."
        mode = _infer_dialogue_mode(question)
        if mode == "compare":
            why_it_fits = "I can compare options for you, but I need to know which tradeoff matters most first."
        elif mode == "confused":
            why_it_fits = "I do not want to force a recommendation before we define what matters most to you."
        else:
            why_it_fits = "I can guide you better once one priority is clear."
        alternative = ""
        if not (next_action or "").strip():
            if mode == "compare":
                next_action = "Should we compare mainly by selectivity, practical outcomes, or affordability?"
            elif mode == "confused":
                next_action = "Would you like to start from city preference or budget comfort?"
            else:
                next_action = "Do you want to prioritize city, budget comfort, or selectivity first?"
        return short_answer, why_it_fits, alternative, next_action

    if not (short_answer or "").strip():
        mode = _infer_dialogue_mode(question)
        if mode == "compare":
            if alt_name and alt_name.lower() != top_name.lower():
                short_answer = f"{opening}. We should compare {top_name} and {alt_name} directly."
            else:
                short_answer = f"{opening}. We should compare the closest options side by side."
        elif mode == "confused":
            if alt_name and alt_name.lower() != top_name.lower():
                short_answer = f"{opening}. We can begin with {top_name} and keep {alt_name} as a backup while we narrow your priorities."
            else:
                short_answer = f"{opening}. We can begin with one practical option and keep a backup while we narrow your priorities."
        elif mode == "vague":
            if alt_name and alt_name.lower() != top_name.lower():
                short_answer = f"{opening}. {top_name} and {alt_name} are both reasonable directions while we refine what matters most to you."
            else:
                short_answer = f"{opening}. There are multiple reasonable directions while we refine what matters most to you."
        else:
            short_answer = f"{opening}. Based on your current profile, {top_name} is a strong first option."

    if not (why_it_fits or "").strip():
        why_it_fits = _compose_why_from_facts(facts=top_facts, profile=profile, question=question)

    if intent in {"compare", "explore"} and alt_name and alt_name.lower() != top_name.lower():
        alternative = _compose_alternative_from_facts(facts=alt_facts, question=question)
    elif not (alternative or "").strip() and alt_name and alt_name.lower() != top_name.lower():
        alternative = _compose_alternative_from_facts(facts=alt_facts, question=question)

    if not (next_action or "").strip():
        mode = _infer_dialogue_mode(question)
        if mode == "confused":
            next_action = "Would it help if we narrow this down with one simple priority first, like city or budget?"
        elif mode == "compare":
            next_action = "Do you want to prioritize selectivity, practical outcomes, or affordability in that comparison?"
        elif mode == "vague":
            next_action = "Are you leaning more toward a technical path or something broader for now?"
        else:
            next_action = "Is that close to what you had in mind, or should we pivot toward another direction?"

    return short_answer, why_it_fits, alternative, next_action


def _build_message_paragraph(
    *,
    short_answer: str,
    why_it_fits: str,
    alternative: str,
    next_action: str,
) -> str:
    def _clean(text: str) -> str:
        return " ".join(str(text or "").strip().split())

    def _limit_words(text: str, max_words: int) -> str:
        words = _clean(text).split()
        if len(words) <= max_words:
            return " ".join(words)
        return " ".join(words[:max_words]).rstrip(".,;: ") + "..."

    def _as_sentence(text: str) -> str:
        t = _clean(text)
        if not t:
            return ""
        if t[-1] not in ".!?":
            t += "."
        return t

    rec = _clean(short_answer)
    fit = _clean(why_it_fits)
    alt = _clean(alternative)
    nxt = _clean(next_action)

    parts: list[str] = []
    if rec:
        parts.append(_as_sentence(_limit_words(rec, 24)))
    if fit:
        parts.append(_as_sentence(_limit_words(fit, 34)))
    if alt:
        parts.append(_as_sentence(_limit_words(alt, 22)))
    if nxt:
        nxt_sentence = _as_sentence(_limit_words(nxt, 18))
        if nxt_sentence:
            parts.append(nxt_sentence)

    sentences: list[str] = []
    seen: set[str] = set()
    for s in parts:
        key = _clean(s).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(s)

    return " ".join(sentences)


def _admission_route_hint(school_name: str, school: dict[str, Any]) -> str:
    name = str(school_name or "").lower()
    selectivity = str(school.get("admission_selectivity", "")).strip().lower()
    if any(k in name for k in ["emi", "ehtp", "ensias", "inpt", "emines"]):
        return "usually through CPGE/CNC path; competitive"
    if "ensa" in name:
        return "often accessible after Bac via selection and exam"
    if "est" in name or "ofppt" in name:
        return "generally more accessible and practical"
    if selectivity:
        return f"selectivity: {selectivity}"
    return "admission route depends on program and yearly seats"


def _school_strength_hint(school: dict[str, Any]) -> str:
    labels = _humanize_programs(_school_program_labels(school))
    if labels:
        return labels
    programs = school.get("programs", [])
    if isinstance(programs, list) and programs:
        cleaned = [" ".join(str(p).replace("_", " ").split()) for p in programs[:4] if str(p).strip()]
        if cleaned:
            return ", ".join(cleaned)
    return "engineering and applied tracks"


def _build_structured_advisor_response(
    *,
    question: str,
    profile: UserProfile,
    ranked_schools: list[dict[str, Any]],
    next_action: str,
) -> str:
    bac_key = (profile.bac_stream or "").strip().lower()
    bac_label = _BAC_LABEL.get(bac_key, bac_key or "bac")
    city_label = profile.city.strip() if (profile.city or "").strip() else "Morocco"
    q_tokens = _norm_tokens(question)

    engineering_terms = {
        "engineering", "ingenierie", "ingenieur", "genie", "cpge", "cpi", "ensa", "est", "emi", "ehtp"
    }
    university_terms = {
        "university", "universite", "fac", "faculty", "fs", "fsjes", "fst"
    }
    business_terms = {
        "business", "management", "finance", "commerce", "encg", "iscae", "marketing"
    }
    health_terms = {"medecine", "medicine", "medical", "health", "paramedical", "sante"}

    path_lines: list[str] = []
    if q_tokens & engineering_terms:
        path_lines.append("- Engineering path (CPGE/CNC or integrated engineering schools)")
        path_lines.append("- Applied path (ENSA/EST/OFPPT style practical route)")
    if q_tokens & business_terms:
        path_lines.append("- Business/management path (commerce, finance, management schools)")
    if q_tokens & health_terms:
        path_lines.append("- Health path (medical or paramedical tracks)")
    if q_tokens & university_terms:
        path_lines.append("- University path (more flexibility and wider program choice)")

    if not path_lines:
        path_lines = [
            "- Ambitious path (selective schools)",
            "- Balanced path (good quality with realistic admission)",
            "- Safe path (accessible options with strong outcomes)",
        ]

    lines: list[str] = []
    lines.append(
        f"Since you are a Bac {bac_label} student in {city_label}, your best choices depend on your target path:"
    )
    lines.extend(path_lines)
    lines.append("")
    lines.append("Here are 3 strong schools to target:")

    top3 = ranked_schools[:3]
    for idx, item in enumerate(top3, start=1):
        name = str(item.get("name", "School"))
        city = str(item.get("city", "")).strip()
        school_header = f"{idx}. {name}" + (f" - {city}" if city else "")
        lines.append(school_header)

        school_payload = {
            "name": item.get("name", ""),
            "programs": item.get("programs", []),
            "programs_tags": item.get("programs_tags", ""),
            "filieres": item.get("filieres", ""),
            "admission_selectivity": item.get("admission_selectivity", ""),
        }
        strengths = _school_strength_hint(school_payload)
        admission = _admission_route_hint(name, school_payload)

        lines.append(f"   - Strong in: {strengths}")
        lines.append(f"   - Admission: {admission}")

        tuition_min = _to_int(item.get("tuition_min_mad"), default=0)
        tuition_max = _to_int(item.get("tuition_max_mad"), default=0)
        if tuition_max > 0:
            lines.append(f"   - Tuition range: {tuition_min}-{tuition_max} MAD")
        lines.append("")

    lines.append("Simple advice for you:")
    if bac_key == "sm":
        if q_tokens & engineering_terms:
            lines.append("- If you are strong in math/physics and accept pressure: prioritize CPGE then selective engineering schools.")
            lines.append("- Keep an ENSA/EST style option as your safer backup.")
        elif q_tokens & business_terms:
            lines.append("- For Bac SM, your math profile is a strong asset for finance/management tracks.")
            lines.append("- Keep one selective commerce school and one balanced backup.")
        else:
            lines.append("- Use your math profile to keep one ambitious option and one realistic backup.")
            lines.append("- Favor schools with clear admission route and program fit.")
    else:
        lines.append("- Start with schools matching your strongest subjects and realistic admission level.")
        lines.append("- Keep one ambitious option, one balanced option, and one safe option.")

    lines.append("")
    lines.append("My recommendation:")
    if top3:
        lines.append(f"- Top target: {top3[0].get('name', 'N/A')}")
    if len(top3) > 1:
        lines.append(f"- Strong backup: {top3[1].get('name', 'N/A')}")
    if len(top3) > 2:
        lines.append(f"- Safe backup: {top3[2].get('name', 'N/A')}")
    lines.append(f"- Next step: {next_action}")

    return "\n".join(lines).strip()


def _match_grade(score_0_100: float) -> str:
    if score_0_100 >= 90:
        return "A+"
    if score_0_100 >= 80:
        return "A"
    if score_0_100 >= 70:
        return "B+"
    if score_0_100 >= 60:
        return "B"
    return "C"


def _augment_question_with_history(question: str, chat_history: list[dict[str, str]] | None) -> str:
    if not chat_history:
        return question

    recent_user_msgs: list[str] = []
    for item in chat_history[-8:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = " ".join(str(item.get("content", "")).split())
        if role == "user" and content:
            recent_user_msgs.append(content)

    if not recent_user_msgs:
        return question

    history_text = " | ".join(recent_user_msgs[-4:])
    return f"{question}\nConversation context: {history_text}"


def _merge_query_understanding_into_request(
    *,
    question: str,
    profile: UserProfile,
    query_understanding: dict[str, Any],
) -> tuple[str, UserProfile]:
    if not query_understanding:
        return question, profile

    merged_question = str(question or "").strip()
    reformulated = " ".join(str(query_understanding.get("reformulated_question", "")).split())
    domains = [str(v).strip().lower() for v in (query_understanding.get("domains") or []) if str(v).strip()]
    excluded_domains = [str(v).strip().lower() for v in (query_understanding.get("excluded_domains") or []) if str(v).strip()]
    city = " ".join(str(query_understanding.get("city", "")).split())

    hints: list[str] = []
    if domains:
        hints.append("domain " + " ".join(domains))
    if excluded_domains:
        hints.append("exclude " + " ".join(excluded_domains))
    if city:
        hints.append(f"city {city}")
    if bool(query_understanding.get("strict_constraints", False)):
        hints.append("strict domain filtering")

    if reformulated and reformulated.lower() not in merged_question.lower():
        merged_question += f"\nReformulated intent: {reformulated}"
    if hints:
        merged_question += "\nStructured constraints: " + "; ".join(hints) + "."

    try:
        confidence = float(query_understanding.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    use_overrides = confidence >= 0.45

    if not use_overrides:
        return merged_question, profile

    bac_stream = str(query_understanding.get("bac_stream", "")).strip().lower()
    budget_band = str(query_understanding.get("budget_band", "")).strip().lower()
    motivation = str(query_understanding.get("motivation", "")).strip().lower()

    bac_override = bac_stream if bac_stream and (_question_mentions_bac(question, bac_stream) or not profile.bac_stream) else ""
    city_override = city if city and (_question_mentions_city(question, city) or not profile.city) else ""
    budget_override = budget_band if budget_band and (_question_mentions_budget(question) or not profile.budget_band) else ""
    motivation_override = motivation if motivation and (_question_mentions_motivation(question) or not profile.motivation) else ""

    merged_profile = UserProfile(
        bac_stream=bac_override or profile.bac_stream,
        expected_grade_band=profile.expected_grade_band,
        motivation=motivation_override or profile.motivation,
        budget_band=budget_override or profile.budget_band,
        city=city_override or profile.city,
        country=profile.country,
        classe=profile.classe,
        note_esperee=profile.note_esperee,
    )
    return merged_question, merged_profile


def answer_question(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
    chat_history: list[dict[str, str]] | None = None,
    user_id: str = "",
    mode: str = "auto",
) -> QueryResponse:
    city_only_mode = False
    is_fr = True
    user_question = " ".join(str(question or "").split())
    response_language = detect_language(user_question)

    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "chat", "recommendation"}:
        normalized_mode = "auto"

    intent = "orientation"
    if normalized_mode == "chat":
        intent = "chat"
    elif normalized_mode == "recommendation":
        intent = "orientation"
    elif user_question:
        try:
            intent = QWEN_GENERATOR.classify_intent(user_question)
        except Exception:
            intent = "orientation"

    if intent == "chat":
        try:
            chat_text = QWEN_GENERATOR.generate_chat_response(
                message=user_question,
                chat_history=chat_history,
                response_language=response_language,
            )
        except Exception:
            chat_text = ""
        _record_output(chat_text)
        return QueryResponse(
            short_answer=chat_text,
            why_it_fits="",
            evidence=[],
            alternative="",
            next_action="",
            confidence=0.85,
            message_paragraph=chat_text,
            ranked_schools=[],
        )

    if not _profile_has_signal(profile):
        return QueryResponse(
            short_answer="On peut faire mieux qu une recommandation au hasard, mais j ai besoin d un peu de contexte sur toi.",
            why_it_fits=(
                "Si tu me donnes juste ton objectif principal, je peux deja te guider de facon utile sans te noyer dans les options."
            ),
            evidence=[],
            alternative="Par exemple, tu peux commencer par me dire la ville que tu preferes et le type de parcours que tu imagines.",
            next_action="Tu veux qu on commence par clarifier ton objectif ou ton budget en premier ?",
            confidence=0.0,
        )

    query_for_context = _profile_to_query(profile)
    is_profile_placeholder = _is_placeholder_profile_request(user_question)
    base_question = query_for_context if is_profile_placeholder else (user_question or query_for_context)

    query_understanding: dict[str, Any] = {}
    if user_question and not is_profile_placeholder:
        try:
            query_understanding = QWEN_GENERATOR.understand_query(
                question=user_question,
                profile=profile,
                chat_history=chat_history,
            )
        except Exception:
            query_understanding = {}

    retrieval_question, retrieval_profile = _merge_query_understanding_into_request(
        question=base_question,
        profile=profile,
        query_understanding=query_understanding,
    )
    retrieval_question = " ".join(str(retrieval_question or "").split()) or base_question

    effective_profile = resolve_effective_profile(
        question=retrieval_question,
        profile=retrieval_profile,
        schools=schools,
    )

    career_profile: dict[str, Any] | None = None
    if user_id:
        try:
            career_profile = fetch_user_career_profile(user_id)
        except Exception:
            career_profile = None

    hits = retrieve(
        question=retrieval_question,
        profile=effective_profile,
        schools=schools,
        transcripts=transcripts,
        top_k=top_k,
        career_profile=career_profile,
    )

    if not hits:
        return QueryResponse(
            short_answer=(
                "Je n ai pas trouve d ecole qui respecte exactement ton profil."
            ),
            why_it_fits=(
                "Les options disponibles ne correspondent pas assez a tes contraintes de budget, niveau, ville, ou orientation."
            ),
            evidence=[],
            alternative=(
                "Tu peux elargir la ville cible ou assouplir la contrainte budget pour obtenir plus de matchs."
            ),
            next_action=(
                "Mets a jour ton profil et je recalculerai le top ecoles uniquement sur ces informations."
            ),
            confidence=0.1,
        )

    evidence: list[EvidenceItem] = []
    for hit in hits:
        school = hit["school"]
        chunk = hit["chunk"]
        evidence.append(
            EvidenceItem(
                chunk_id=chunk["chunk_id"],
                school_id=chunk["school_id"],
                school_name=school["name"],
                program=chunk["program"],
                recorded_at=chunk["recorded_at"],
                text=chunk["text"],
                score=round(float(hit["score"]), 4),
            )
        )

    top_schools: list[dict] = []
    ranked_schools: list[dict[str, Any]] = []
    for hit in hits[:5]:
        school = hit["school"]
        components = hit.get("score_components", {})
        distance_km_raw = components.get("distance_km")
        distance_km = None
        if distance_km_raw is not None:
            try:
                distance_km = round(float(distance_km_raw), 1)
            except (TypeError, ValueError):
                distance_km = None
        profile_priority = float(components.get("profile_priority", 0.0))
        career_domain_match = float(components.get("career_domain_match", 0.0))
        blended = 0.8 * profile_priority + 0.2 * career_domain_match
        if blended <= 0.0:
            blended = (
                0.5 * float(components.get("bac_semantic", 0.0))
                + 0.2 * float(components.get("location_match", 0.0))
                + 0.15 * float(components.get("budget_match", 0.0))
                + 0.15 * float(components.get("motivation_match", 0.0))
            )
        match_score = round(100.0 * blended, 1)
        match_grade = _match_grade(match_score)
        top_schools.append(
            {
                "school_id": school.get("school_id"),
                "name": school.get("name"),
                "city": school.get("city"),
                "programs": school.get("programs", []),
                "tuition_min_mad": school.get("tuition_min_mad"),
                "tuition_max_mad": school.get("tuition_max_mad"),
                "pricing_min": school.get("pricing_min"),
                "pricing_max": school.get("pricing_max"),
                "legal_status": school.get("legal_status"),
                "website_url": school.get("website_url"),
                "programs_tags": school.get("programs_tags"),
                "filieres": school.get("filieres"),
                "admission_selectivity": school.get("admission_selectivity"),
                "score": round(float(hit.get("score", 0.0)), 4),
                "match_score": match_score,
                "match_grade": match_grade,
                "distance_km": distance_km,
                "score_components": {
                    "program_match": round(float(components.get("program_match", 0.0)), 4),
                    "budget_match": round(float(components.get("budget_match", 0.0)), 4),
                    "grade_match": round(float(components.get("grade_match", 0.0)), 4),
                    "location_match": round(float(components.get("location_match", 0.0)), 4),
                    "motivation_match": round(float(components.get("motivation_match", 0.0)), 4),
                    "bac_semantic": round(float(components.get("bac_semantic", 0.0)), 4),
                    "weighted": round(float(components.get("weighted", 0.0)), 4),
                    "profile_priority": round(float(components.get("profile_priority", 0.0)), 4),
                    "career_domain_match": round(float(components.get("career_domain_match", 0.0)), 4),
                    "career_overlap": round(float(components.get("career_overlap", 0.0)), 4),
                    "domain_alignment": round(float(components.get("domain_alignment", 0.0)), 4),
                    "profile_constraints_match": round(float(components.get("profile_constraints_match", 0.0)), 4),
                    "public_constraints_match": round(float(components.get("public_constraints_match", 0.0)), 4),
                },
            }
        )
        ranked_schools.append(
            {
                "school_id": school.get("school_id"),
                "name": school.get("name"),
                "city": school.get("city"),
                "match_score": match_score,
                "match_grade": match_grade,
                "distance_km": distance_km,
                "criteria": {
                    "semantic_fit": round(100.0 * float(components.get("bac_semantic", 0.0)), 1),
                    "geo_fit": round(100.0 * float(components.get("location_match", 0.0)), 1),
                    "budget_fit": round(100.0 * float(components.get("budget_match", 0.0)), 1),
                    "motivation_fit": round(100.0 * float(components.get("motivation_match", 0.0)), 1),
                },
            }
        )

    top_schools.sort(
        key=lambda item: (
            -float(item.get("match_score", 0.0)),
            float(item.get("distance_km")) if item.get("distance_km") is not None else float("inf"),
        )
    )
    ranked_schools.sort(
        key=lambda item: (
            -float(item.get("match_score", 0.0)),
            float(item.get("distance_km")) if item.get("distance_km") is not None else float("inf"),
        )
    )
    if ranked_schools:
        rank_by_school_id = {
            str(item.get("school_id", "")): idx
            for idx, item in enumerate(ranked_schools)
        }
        hits = sorted(
            hits,
            key=lambda hit: rank_by_school_id.get(
                str(hit.get("school", {}).get("school_id", "")),
                10**6,
            ),
        )
        evidence.sort(
            key=lambda ev: rank_by_school_id.get(str(ev.school_id), 10**6),
        )

    generation_evidence = _select_generation_evidence(evidence, max_items=3)

    rejected_school = _extract_rejected_school_from_history(
        chat_history=chat_history,
        candidate_schools=top_schools,
    )
    if rejected_school:
        rejected_id = str(rejected_school.get("school_id", "")).strip()
        filtered_top = [
            s for s in top_schools
            if str(s.get("school_id", "")).strip() != rejected_id and not _schools_share_direction(s, rejected_school)
        ]
        if filtered_top:
            allowed_ids = {str(s.get("school_id", "")).strip() for s in filtered_top}
            top_schools = filtered_top
            ranked_schools = [s for s in ranked_schools if str(s.get("school_id", "")).strip() in allowed_ids]
            hits = [h for h in hits if str(h.get("school", {}).get("school_id", "")).strip() in allowed_ids]
            evidence = [e for e in evidence if str(e.school_id).strip() in allowed_ids]
            generation_evidence = _select_generation_evidence(evidence, max_items=3)

    detected_city = _extract_detected_city(user_question, top_schools)
    if detected_city:
        filtered_top = [s for s in top_schools if _match_city(str(s.get("city", "")), detected_city)]
        if filtered_top:
            allowed_ids = {str(s.get("school_id", "")).strip() for s in filtered_top}
            top_schools = filtered_top
            ranked_schools = [s for s in ranked_schools if str(s.get("school_id", "")).strip() in allowed_ids]
            hits = [h for h in hits if str(h.get("school", {}).get("school_id", "")).strip() in allowed_ids]
            evidence = [e for e in evidence if str(e.school_id).strip() in allowed_ids]
            generation_evidence = _select_generation_evidence(evidence, max_items=3)

    top_school = top_schools[0] if top_schools else hits[0].get("school", {})
    top_facts = build_school_facts(top_school)
    mode = _infer_dialogue_mode(user_question)
    intent_mode = _intent_from_mode(mode)
    top_score = float(top_schools[0].get("match_score", 0.0)) if top_schools else 0.0
    second_score = float(top_schools[1].get("match_score", 0.0)) if len(top_schools) > 1 else 0.0
    top_gap = max(0.0, top_score - second_score)
    confidence_ratio = max(0.0, min(1.0, top_score / 100.0))
    style = _select_style(mode=mode, confidence=confidence_ratio, top_gap=top_gap, has_school=bool(top_schools))
    suggest_school = style != "clarification_mode"

    if top_schools:
        if intent_mode in {"compare", "explore", "confused"}:
            selected_schools = top_schools[:2]
        else:
            selected_schools = top_schools[:1]
    else:
        selected_schools = []

    if selected_schools:
        top_school = selected_schools[0]

    short_answer = ""
    why_it_fits = ""

    alt_school_obj: dict[str, Any] | None = None
    if len(selected_schools) > 1:
        alt_school_obj = selected_schools[1]
    else:
        alt_hit = _select_alternative_hit(
            hits=hits,
            question=user_question,
            profile=effective_profile,
            rejected_school=rejected_school,
        )
        if alt_hit is not None:
            alt_school_obj = alt_hit.get("school", {})
        elif len(generation_evidence) > 1:
            alt_school_name = str(generation_evidence[1].school_name).strip().lower()
            for item in top_schools[1:]:
                if str(item.get("name", "")).strip().lower() == alt_school_name:
                    alt_school_obj = item
                    break
        else:
            alt_school_obj = top_school

    alternative = ""
    top_website = str(top_school.get("website_url", "")).strip()
    next_action = ""
    if suggest_school and top_website:
        next_action = f"You can also verify details on {top_website}, then we can choose based on your priority."

    if _is_city_only_school_request(user_question, effective_profile):
        city_only_mode = True
        options: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            school = hit.get("school", {})
            school_city = str(school.get("city", "")).strip()
            if effective_profile.city and not _match_city(school_city, effective_profile.city):
                # For broad city intent, only surface schools clearly tied to requested city.
                continue
            name = str(school.get("name", "")).strip()
            if name and name not in seen:
                seen.add(name)
                options.append(name)
            if len(options) >= 3:
                break

        if not options:
            for item in evidence:
                name = item.school_name.strip()
                if name and name not in seen:
                    seen.add(name)
                    options.append(name)
                if len(options) >= 3:
                    break

        city = effective_profile.city or "that city"
        if options:
            if is_fr:
                joined = ", ".join(options[:2]) if len(options) > 1 else options[0]
                short_answer = f"Si ton critere principal est {city}, je commencerais par {joined} puis on affine selon ton objectif."
            else:
                joined = ", ".join(options[:2]) if len(options) > 1 else options[0]
                short_answer = f"If city is your main filter for {city}, I would begin with {joined} and then narrow based on your direction."
        else:
            if is_fr:
                short_answer = f"A {city}, il y a des options interessantes; l important est de choisir celle qui colle a ton objectif reel."
            else:
                short_answer = f"In {city}, there are viable options; the key is matching one to your real goal."

        if is_fr:
            why_it_fits = (
                f"Comme ta demande est centree sur {city}, on peut avancer simplement et choisir selon ton projet avant de comparer plus large."
            )
            alternative = (
                "Si tu veux, on peut aussi regarder une option plus ambitieuse en parallele pour garder un plan B motive."
            )
            next_action = "Tu preferes qu on tranche d abord par type de parcours ou par niveau de selectivite ?"
        else:
            why_it_fits = (
                f"Since you are city-first around {city}, we can keep this simple and decide based on direction before widening the scope."
            )
            alternative = (
                "If you want balance, we can keep one ambitious option in view while staying realistic on your main path."
            )
            next_action = "Do you want to narrow first by learning style, budget comfort, or selectivity level?"

    if _has_conflicting_constraints(user_question, effective_profile):
        if is_fr:
            alternative = (
                "Ta demande combine un budget serre avec un objectif de prestige eleve. Le plus prudent est de commencer par des options publiques abordables, "
                "puis de comparer une option plus selective en choix ambitieux."
            )
        else:
            alternative = (
                "Your request mixes a tight budget with high-prestige goals, so the safest path is to shortlist affordable public options first, "
                "then compare one higher-selectivity option as a stretch choice."
            )

    if not city_only_mode:
        short_answer, why_it_fits, alternative, next_action = _enforce_grounded_response(
            short_answer=short_answer,
            why_it_fits=why_it_fits,
            alternative=alternative,
            next_action=next_action,
            top_school=top_school,
            alt_school=alt_school_obj,
            profile=effective_profile,
            question=user_question,
            style=style,
            suggest_school=suggest_school,
        )

    if intent_mode == "explore" and len(selected_schools) > 1:
        a_name = str(selected_schools[0].get("name", "")).strip()
        b_name = str(selected_schools[1].get("name", "")).strip()
        short_answer = f"{_pick_opening(style)}. {a_name} and {b_name} both look viable depending on whether you want a more selective path or a more accessible one."
        why_it_fits = "At this stage, it is better to compare fit factors than to push a single best choice."
        alternative = ""
        if not (next_action or "").strip():
            next_action = "Do you want to narrow this by budget, selectivity, or program style first?"

    assert "score=" not in why_it_fits
    assert "chunk=" not in why_it_fits
    assert "sb_" not in why_it_fits

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))
    payload = {
        "short_answer": short_answer,
        "why_it_fits": why_it_fits,
        "alternative": alternative,
        "next_action": next_action,
    }
    rewrite_facts = {
        "top_school": top_facts,
        "allowed_school_names": [str(item.get("name", "")).strip() for item in top_schools if str(item.get("name", "")).strip()],
        "allowed_cities": [str(item.get("city", "")).strip() for item in top_schools if str(item.get("city", "")).strip()],
    }
    message_paragraph = ""
    try:
        message_paragraph = QWEN_GENERATOR.rewrite_to_natural_response(
            payload=payload,
            question=user_question or query_for_context,
            facts=rewrite_facts,
            response_language=response_language,
            reframe_instruction=f"style_seed={style}; avoid opening='{', '.join(_recent_openings(last_n=3))}'",
        )
    except Exception:
        message_paragraph = ""

    message_paragraph = _sanitize_user_facing_text(message_paragraph)
    if _looks_structured_output(message_paragraph) or _opening_sentence(message_paragraph) in _recent_openings(last_n=3):
        alt_style = _alternate_style(style)
        short_answer, why_it_fits, alternative, next_action = _enforce_grounded_response(
            short_answer="",
            why_it_fits="",
            alternative="",
            next_action="",
            top_school=top_school,
            alt_school=alt_school_obj,
            profile=effective_profile,
            question=user_question,
            style=alt_style,
            suggest_school=suggest_school,
        )
        payload = {
            "short_answer": short_answer,
            "why_it_fits": why_it_fits,
            "alternative": alternative,
            "next_action": next_action,
        }
        try:
            message_paragraph = QWEN_GENERATOR.rewrite_to_natural_response(
                payload=payload,
                question=user_question or query_for_context,
                facts=rewrite_facts,
                response_language=response_language,
                reframe_instruction=f"style_seed={alt_style}; avoid opening='{', '.join(_recent_openings(last_n=3))}'",
            )
        except Exception:
            message_paragraph = ""
        message_paragraph = _sanitize_user_facing_text(message_paragraph)

    if not message_paragraph or _looks_structured_output(message_paragraph):
        message_paragraph = _sanitize_user_facing_text(f"{short_answer} {next_action}")
    if _opening_sentence(message_paragraph) in _recent_openings(last_n=3):
        message_paragraph = _sanitize_user_facing_text(f"{_pick_opening(style)}. {next_action}")
    _record_output(message_paragraph)

    return QueryResponse(
        short_answer=short_answer,
        why_it_fits=why_it_fits,
        evidence=evidence,
        alternative=alternative,
        next_action=next_action,
        confidence=round(confidence, 3),
        message_paragraph=message_paragraph,
        ranked_schools=ranked_schools,
    )
