from __future__ import annotations

import re
from statistics import mean
from typing import Any

from app.generator import QWEN_GENERATOR
from app.models import EvidenceItem, QueryResponse, UserProfile
from app.retriever import resolve_effective_profile, retrieve


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


def _grounding_ratio(text: str, evidence_tokens: set[str]) -> float:
    tokens = _content_tokens(text)
    if not tokens:
        return 1.0
    return len(tokens & evidence_tokens) / float(len(tokens))


def _excerpt(text: str, words: int) -> str:
    return " ".join(str(text or "").split()[:words])


def _enforce_grounded_response(
    *,
    short_answer: str,
    why_it_fits: str,
    alternative: str,
    next_action: str,
    generation_evidence: list[EvidenceItem],
    profile: UserProfile,
) -> tuple[str, str, str, str]:
    if not generation_evidence:
        return short_answer, why_it_fits, alternative, next_action

    top_ev = generation_evidence[0]
    alt_ev = generation_evidence[1] if len(generation_evidence) > 1 else top_ev
    city_hint = f" in {profile.city}" if profile.city else ""

    evidence_tokens: set[str] = set()
    for ev in generation_evidence:
        evidence_tokens |= _content_tokens(f"{ev.school_name} {ev.program} {ev.text}")

    if _grounding_ratio(short_answer, evidence_tokens) < 0.18:
        short_answer = f"Based on the retrieved evidence, {top_ev.school_name} looks like the strongest match{city_hint}."

    if _grounding_ratio(why_it_fits, evidence_tokens) < 0.18:
        why_it_fits = f"Evidence for {top_ev.school_name}{city_hint}: {_excerpt(top_ev.text, 24)}."

    if _grounding_ratio(alternative, evidence_tokens) < 0.15:
        alternative = f"A grounded alternative is {alt_ev.school_name}, with supporting details: {_excerpt(alt_ev.text, 20)}."

    if not (next_action or "").strip():
        next_action = "Share your target program, budget range, and preferred city so I can narrow this further."

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
) -> QueryResponse:
    city_only_mode = False
    is_fr = True

    if not _profile_has_signal(profile):
        return QueryResponse(
            short_answer="Donne moi ton profil pour que je recommande les meilleures ecoles.",
            why_it_fits=(
                "Je base les recommandations uniquement sur ton profil: filiere bac, budget, motivation, ville et niveau attendu."
            ),
            evidence=[],
            alternative=(
                "Exemple de profil utile: bac=spc, budget=tight_25k, motivation=employability, city=Rabat."
            ),
            next_action=(
                "Envoie ton profil et je te renvoie directement un classement des ecoles les plus compatibles."
            ),
            confidence=0.0,
        )

    query_for_context = _profile_to_query(profile)
    retrieval_question = query_for_context
    retrieval_profile = profile
    question = retrieval_question

    effective_profile = resolve_effective_profile(
        question=retrieval_question,
        profile=retrieval_profile,
        schools=schools,
    )

    hits = retrieve(
        question=retrieval_question,
        profile=effective_profile,
        schools=schools,
        transcripts=transcripts,
        top_k=top_k,
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
        match_score = round(
            100.0
            * (
                0.5 * float(components.get("bac_semantic", 0.0))
                + 0.2 * float(components.get("location_match", 0.0))
                + 0.15 * float(components.get("budget_match", 0.0))
                + 0.15 * float(components.get("motivation_match", 0.0))
            ),
            1,
        )
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
            float(item.get("distance_km")) if item.get("distance_km") is not None else float("inf"),
            -float(item.get("match_score", 0.0)),
        )
    )
    ranked_schools.sort(
        key=lambda item: (
            float(item.get("distance_km")) if item.get("distance_km") is not None else float("inf"),
            -float(item.get("match_score", 0.0)),
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
    top_ev = generation_evidence[0]
    top_school = top_schools[0] if top_schools else hits[0].get("school", {})
    top_status = _school_status_text(top_school)
    top_min, top_max = _school_tuition_range(top_school)
    tuition_text = f" with tuition around {top_min}-{top_max} MAD" if top_max > 0 else ""
    short_answer = f"I’d start with {top_ev.school_name} for what you asked"
    if top_status:
        short_answer += f", especially since it is {top_status.lower()}"
    short_answer += f"{tuition_text}."

    ev_text = " ".join(str(top_ev.text).split())
    ev_excerpt = " ".join(ev_text.split()[:24])
    city_hint = f" in {effective_profile.city}" if effective_profile.city else ""
    top_programs = _school_program_labels(top_school)
    top_programs_human = _humanize_programs(top_programs)
    program_hint = f" Key program areas include {top_programs_human}." if top_programs_human else ""
    evidence_hint = f" Retrieved evidence highlights: {ev_excerpt}." if ev_excerpt else ""
    why_it_fits = f"It fits your request{city_hint} in a practical way.{program_hint}{evidence_hint}".strip()

    alt_hit = _select_alternative_hit(
        hits=hits,
        question=question,
        profile=effective_profile,
    )
    if alt_hit is not None:
        alt_school_obj = alt_hit.get("school", {})
        alt_school = str(alt_school_obj.get("name", top_ev.school_name))
        alt_min, alt_max = _school_tuition_range(alt_school_obj)
        alt_programs = _humanize_programs(_school_program_labels(alt_school_obj))
        alt_bits: list[str] = []
        if alt_programs:
            alt_bits.append(f"program focus: {alt_programs}")
        if alt_max > 0:
            alt_bits.append(f"tuition around {alt_min}-{alt_max} MAD")
        alt_excerpt = ", ".join(alt_bits) if alt_bits else "a profile close to your criteria"
    elif len(generation_evidence) > 1:
        alt_school = generation_evidence[1].school_name
        alt_text = " ".join(str(generation_evidence[1].text).split())
        alt_excerpt = " ".join(alt_text.split()[:16]) if alt_text else "a profile close to your criteria"
    else:
        alt_school = top_ev.school_name
        alt_excerpt = "a profile close to your criteria"

    alternative = f"Another option worth checking is {alt_school}, especially for {alt_excerpt}."
    top_website = str(top_school.get("website_url", "")).strip()
    next_action = "If you share your target program, budget range, and preferred study duration, I can make it more precise."
    if top_website:
        next_action = f"You can verify official details on {top_website}. Then tell me your target program and budget so I can narrow the shortlist."

    try:
        generated = QWEN_GENERATOR.generate(
            question=query_for_context,
            profile=effective_profile,
            top_schools=top_schools,
            generation_evidence=generation_evidence,
        )
    except Exception:
        generated = {}

    if generated:
        short_answer = str(generated.get("short_answer", short_answer)).strip() or short_answer
        why_it_fits = str(generated.get("why_it_fits", why_it_fits)).strip() or why_it_fits
        alternative = str(generated.get("alternative", alternative)).strip() or alternative
        next_action = str(generated.get("next_action", next_action)).strip() or next_action

    if _is_city_only_school_request(question, effective_profile):
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
                joined = ", ".join(options[:-1]) + (f", et {options[-1]}" if len(options) > 1 else options[0])
                short_answer = f"Bon choix. A {city}, je te propose de commencer par {joined}."
            else:
                joined = ", ".join(options[:-1]) + (f", and {options[-1]}" if len(options) > 1 else options[0])
                short_answer = f"Good choice. In {city}, I’d start with options like {joined}."
        else:
            if is_fr:
                short_answer = f"Bon choix. Il existe quelques options solides a {city}."
            else:
                short_answer = f"Good choice. There are a few solid school options in {city}."

        if is_fr:
            why_it_fits = (
                f"Comme ta demande est centree sur la ville, voici une shortlist simple pour {city}. "
                "Le meilleur choix dependra de la filiere, du budget, et du niveau attendu."
            )
            alternative = (
                "Une alternative pratique est de commencer par des parcours publics ou professionnalisants si le cout est prioritaire, puis comparer une option plus selective."
            )
            next_action = "Donne moi la filiere visee, le budget, et la note attendue pour une recommandation precise."
        else:
            why_it_fits = (
                f"Because your request is city-only, this is a simple shortlist for {city}. "
                "The best fit will depend on your field, budget, and expected grade."
            )
            alternative = (
                "A practical alternative is to begin with public or vocational tracks if affordability matters most, then compare one selective option too."
            )
            next_action = "Tell me your intended field, your budget band, and your expected grade so I can give one precise recommendation."

    if _has_conflicting_constraints(question, effective_profile):
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
            generation_evidence=generation_evidence,
            profile=effective_profile,
        )

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))
    message_paragraph = _build_message_paragraph(
        short_answer=short_answer,
        why_it_fits=why_it_fits,
        alternative=alternative,
        next_action=next_action,
    )

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
