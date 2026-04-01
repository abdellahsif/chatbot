from __future__ import annotations

import re
from statistics import mean
from typing import Any

from app.models import EvidenceItem, QueryResponse, UserProfile
from app.retriever import resolve_effective_profile, retrieve


_BUDGET_MAX = {
    "zero_public": 0,
    "tight_25k": 25000,
    "comfort_50k": 50000,
    "no_limit_70k_plus": 10**9,
}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _norm_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _match_city(city_a: str, city_b: str) -> bool:
    a = " ".join((city_a or "").strip().lower().split())
    b = " ".join((city_b or "").strip().lower().split())
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _select_alternative_hit(
    *,
    hits: list[dict],
    question: str,
    profile: UserProfile,
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

        chunk = hit.get("chunk", {})
        name = str(school.get("name", ""))
        school_type = str(school.get("type", "")).lower()
        school_city = str(school.get("city", ""))
        programs = " ".join(school.get("programs", []))
        chunk_text = str(chunk.get("text", ""))
        school_tokens = _norm_tokens(f"{name} {school_type} {programs} {chunk.get('program', '')} {chunk_text}")
        tuition_max = _to_int(school.get("tuition_max_mad"), default=10**9)
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

        if top_is_public and "public" in school_type:
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


def answer_question(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> QueryResponse:
    if not (question or "").strip():
        return QueryResponse(
            short_answer="Please provide your question.",
            why_it_fits="I need your target program, city, and budget to recommend a school.",
            evidence=[],
            alternative="Example: 'Computer science in Rabat with medium budget'.",
            next_action="Tell me your exact study goal and constraints.",
            confidence=0.0,
        )

    if _is_greeting_or_low_intent(question):
        return QueryResponse(
            short_answer="Hi! I can help you choose a school in Morocco.",
            why_it_fits="Your message looks like a greeting, so I need more details before recommending schools.",
            evidence=[],
            alternative="Example 1: 'Computer science school in Rabat with medium budget'.",
            next_action="Tell me program, city, and budget band to start.",
            confidence=0.0,
        )

    effective_profile = resolve_effective_profile(
        question=question,
        profile=profile,
        schools=schools,
    )

    hits = retrieve(
        question=question,
        profile=effective_profile,
        schools=schools,
        transcripts=transcripts,
        top_k=top_k,
    )

    if not hits:
        return QueryResponse(
            short_answer="No suitable school found for your constraints.",
            why_it_fits="No candidate passed budget/bac/country filtering. Try widening budget or changing city.",
            evidence=[],
            alternative="Try public schools or a nearby city with lower tuition.",
            next_action="Tell me your exact target program and acceptable budget range.",
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
    for hit in hits[:5]:
        school = hit["school"]
        components = hit.get("score_components", {})
        top_schools.append(
            {
                "school_id": school.get("school_id"),
                "name": school.get("name"),
                "city": school.get("city"),
                "programs": school.get("programs", []),
                "tuition_min_mad": school.get("tuition_min_mad"),
                "tuition_max_mad": school.get("tuition_max_mad"),
                "admission_selectivity": school.get("admission_selectivity"),
                "score": round(float(hit.get("score", 0.0)), 4),
                "score_components": {
                    "program_match": round(float(components.get("program_match", 0.0)), 4),
                    "budget_match": round(float(components.get("budget_match", 0.0)), 4),
                    "grade_match": round(float(components.get("grade_match", 0.0)), 4),
                    "location_match": round(float(components.get("location_match", 0.0)), 4),
                    "motivation_match": round(float(components.get("motivation_match", 0.0)), 4),
                    "weighted": round(float(components.get("weighted", 0.0)), 4),
                },
            }
        )

    top_ev = evidence[0]
    short_answer = f"{top_ev.school_name} looks like the strongest match for what you asked."

    ev_text = " ".join(str(top_ev.text).split())
    ev_excerpt = " ".join(ev_text.split()[:28])
    why_it_fits = f"It aligns with your request based on this evidence: {ev_excerpt}."

    alt_hit = _select_alternative_hit(
        hits=hits,
        question=question,
        profile=effective_profile,
    )
    if alt_hit is not None:
        alt_school = str(alt_hit.get("school", {}).get("name", top_ev.school_name))
        alt_text = " ".join(str(alt_hit.get("chunk", {}).get("text", "")).split())
        alt_excerpt = " ".join(alt_text.split()[:20]) if alt_text else ev_excerpt
    elif len(evidence) > 1:
        alt_school = evidence[1].school_name
        alt_text = " ".join(str(evidence[1].text).split())
        alt_excerpt = " ".join(alt_text.split()[:20])
    else:
        alt_school = top_ev.school_name
        alt_excerpt = ev_excerpt

    alternative = f"A solid alternative is {alt_school}, with supporting details: {alt_excerpt}."
    next_action = "If you share your target program, budget range, and preferred study duration, I can narrow this to a sharper shortlist."

    if _is_city_only_school_request(question, effective_profile):
        options: list[str] = []
        seen: set[str] = set()
        for item in evidence:
            name = item.school_name.strip()
            if name and name not in seen:
                seen.add(name)
                options.append(name)
            if len(options) >= 3:
                break

        city = effective_profile.city or "that city"
        if options:
            joined = ", ".join(options[:-1]) + (f", and {options[-1]}" if len(options) > 1 else options[0])
            short_answer = f"Good choice. In {city}, you can start with options like {joined}."
        else:
            short_answer = f"Good choice. There are multiple school options in {city}."

        why_it_fits = "The best option depends on your field, budget, and grade level, so a broad city-only request works better as an initial shortlist."
        alternative = "If you want practical and lower-cost outcomes, vocational or public tracks are usually the safest first filter."
        next_action = "Tell me your intended field (for example IT, business, or health), your budget band, and your expected grade so I can give one precise recommendation."

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))

    return QueryResponse(
        short_answer=short_answer,
        why_it_fits=why_it_fits,
        evidence=evidence,
        alternative=alternative,
        next_action=next_action,
        confidence=round(confidence, 3),
    )
