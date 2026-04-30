from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable

from app.models import EvidenceItem, UserProfile
from app.retriever import resolve_effective_profile, retrieve
from app.supabase_store import fetch_user_career_profile


QueryUnderstandingProvider = Callable[..., dict[str, Any]]

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


@dataclass
class RecommendationResult:
    query_for_context: str
    retrieval_question: str
    effective_profile: UserProfile
    hits: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    generation_evidence: list[EvidenceItem]
    top_schools: list[dict[str, Any]]
    ranked_schools: list[dict[str, Any]]
    rejected_school: dict[str, Any] | None = None
    detected_city: str = ""
    query_understanding: dict[str, Any] | None = None
    career_profile: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_for_context": self.query_for_context,
            "retrieval_question": self.retrieval_question,
            "effective_profile": asdict(self.effective_profile),
            "top_schools": self.top_schools,
            "ranked_schools": self.ranked_schools,
            "evidence": [asdict(item) for item in self.evidence],
            "detected_city": self.detected_city,
            "query_understanding": self.query_understanding or {},
        }


def has_profile_signal(profile: UserProfile) -> bool:
    return any(
        [
            bool((profile.bac_stream or "").strip()),
            bool((profile.expected_grade_band or "").strip()),
            bool((profile.motivation or "").strip()),
            bool((profile.budget_band or "").strip()),
            bool((profile.city or "").strip()),
        ]
    )


def is_placeholder_recommendation_request(question: str) -> bool:
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


def profile_to_retrieval_query(profile: UserProfile) -> str:
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


def _program_tokens_from_school(school: dict[str, Any]) -> set[str]:
    programs = school.get("programs", [])
    program_text = " ".join(str(p) for p in programs if str(p).strip()) if isinstance(programs, list) else str(programs)
    text = " ".join(
        [
            str(school.get("programs_tags", "")),
            str(school.get("filieres", "")),
            program_text,
        ]
    )
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    stop = {
        "de",
        "la",
        "le",
        "et",
        "des",
        "du",
        "the",
        "and",
        "of",
        "program",
        "programs",
        "programme",
        "programmes",
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
        r"\b(je n aime pas|j aime pas|je ne veux pas|pas interesse|autre chose|une autre option|changer de voie)\b",
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


def _question_mentions_city(question: str, city: str) -> bool:
    q = " ".join(str(question or "").lower().split())
    c = " ".join(str(city or "").lower().split())
    return bool(q and c and c in q)


def _question_mentions_bac(question: str, bac_stream: str) -> bool:
    q = " ".join(str(question or "").lower().split())
    b = " ".join(str(bac_stream or "").lower().replace("_", " ").split())
    if not q or not b:
        return False
    return b in q or "bac" in q


def _question_mentions_budget(question: str) -> bool:
    q = " ".join(str(question or "").lower().split())
    return bool(re.search(r"\b(budget|prix|cout|frais|gratuit|public|mad|dh|dirham|money|cost)\b", q))


def _question_mentions_motivation(question: str) -> bool:
    q = " ".join(str(question or "").lower().split())
    return bool(
        re.search(
            r"\b(motivation|objectif|goal|career|emploi|job|prestige|passion|international|salaire|salary|safety|stable)\b",
            q,
        )
    )


def _merge_query_understanding_into_request(
    *,
    question: str,
    profile: UserProfile,
    query_understanding: dict[str, Any],
) -> tuple[str, UserProfile]:
    if not query_understanding:
        return question, profile

    merged_question = question
    reformulated = str(query_understanding.get("reformulated_question", "")).strip()
    domains = [
        str(item).strip()
        for item in query_understanding.get("domains", [])
        if str(item).strip()
    ]
    excluded_domains = [
        str(item).strip()
        for item in query_understanding.get("excluded_domains", [])
        if str(item).strip()
    ]
    city = str(query_understanding.get("city", "")).strip()

    hints: list[str] = []
    if domains:
        hints.append("domains " + ", ".join(domains))
    if excluded_domains:
        hints.append("exclude domains " + ", ".join(excluded_domains))
    if city:
        hints.append(f"city {city}")

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


def _extract_detected_city(question: str, schools: list[dict[str, Any]]) -> str:
    q = " ".join(str(question or "").split()).strip().lower()
    if not q:
        return ""

    for school in schools:
        city = " ".join(str(school.get("city", "")).split()).strip()
        if city and city.lower() in q:
            return city
    match = re.search(r"\b(?:in|at|a|au|aux|en)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", q, flags=re.IGNORECASE)
    if not match:
        return ""
    return " ".join(match.group(1).split()[:3]).strip()


def _match_grade(score_0_100: float) -> str:
    if score_0_100 >= 82:
        return "excellent"
    if score_0_100 >= 65:
        return "strong"
    if score_0_100 >= 45:
        return "possible"
    return "weak"


def _select_generation_evidence(evidence: list[EvidenceItem], max_items: int = 3) -> list[EvidenceItem]:
    selected: list[EvidenceItem] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.school_id.strip().lower() or item.school_name.strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _build_evidence(hits: list[dict[str, Any]]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for hit in hits:
        school = hit.get("school", {})
        chunk = hit.get("chunk", {})
        evidence.append(
            EvidenceItem(
                chunk_id=str(chunk.get("chunk_id", "")),
                school_id=str(chunk.get("school_id", school.get("school_id", ""))),
                school_name=str(school.get("name", "")),
                program=str(chunk.get("program", "")),
                recorded_at=str(chunk.get("recorded_at", "")),
                text=str(chunk.get("text", "")),
                score=round(float(hit.get("score", 0.0)), 4),
            )
        )
    return evidence


def _school_rank_payloads(
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    top_schools: list[dict[str, Any]] = []
    ranked_schools: list[dict[str, Any]] = []
    presentation_k = min(max(1, top_k), 10)

    for hit in hits[:presentation_k]:
        school = hit.get("school", {})
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

    sort_key = lambda item: (
        -float(item.get("match_score", 0.0)),
        float(item.get("distance_km")) if item.get("distance_km") is not None else float("inf"),
    )
    top_schools.sort(key=sort_key)
    ranked_schools.sort(key=sort_key)
    return top_schools, ranked_schools


def _align_hits_and_evidence_to_rank(
    *,
    hits: list[dict[str, Any]],
    evidence: list[EvidenceItem],
    ranked_schools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[EvidenceItem]]:
    if not ranked_schools:
        return hits, evidence

    rank_by_school_id = {
        str(item.get("school_id", "")): idx
        for idx, item in enumerate(ranked_schools)
    }
    sorted_hits = sorted(
        hits,
        key=lambda hit: rank_by_school_id.get(
            str(hit.get("school", {}).get("school_id", "")),
            10**6,
        ),
    )
    sorted_evidence = sorted(
        evidence,
        key=lambda ev: rank_by_school_id.get(str(ev.school_id), 10**6),
    )
    return sorted_hits, sorted_evidence


def _apply_context_filters(
    *,
    question: str,
    chat_history: list[dict[str, str]] | None,
    hits: list[dict[str, Any]],
    evidence: list[EvidenceItem],
    top_schools: list[dict[str, Any]],
    ranked_schools: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[EvidenceItem],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
    str,
]:
    rejected_school = _extract_rejected_school_from_history(
        chat_history=chat_history,
        candidate_schools=top_schools,
    )
    if rejected_school:
        rejected_id = str(rejected_school.get("school_id", "")).strip()
        filtered_top = [
            school
            for school in top_schools
            if str(school.get("school_id", "")).strip() != rejected_id
            and not _schools_share_direction(school, rejected_school)
        ]
        if filtered_top:
            allowed_ids = {str(school.get("school_id", "")).strip() for school in filtered_top}
            top_schools = filtered_top
            ranked_schools = [
                school
                for school in ranked_schools
                if str(school.get("school_id", "")).strip() in allowed_ids
            ]
            hits = [
                hit
                for hit in hits
                if str(hit.get("school", {}).get("school_id", "")).strip() in allowed_ids
            ]
            evidence = [item for item in evidence if str(item.school_id).strip() in allowed_ids]

    detected_city = _extract_detected_city(question, top_schools)
    if detected_city:
        filtered_top = [
            school
            for school in top_schools
            if _match_city(str(school.get("city", "")), detected_city)
        ]
        if filtered_top:
            allowed_ids = {str(school.get("school_id", "")).strip() for school in filtered_top}
            top_schools = filtered_top
            ranked_schools = [
                school
                for school in ranked_schools
                if str(school.get("school_id", "")).strip() in allowed_ids
            ]
            hits = [
                hit
                for hit in hits
                if str(hit.get("school", {}).get("school_id", "")).strip() in allowed_ids
            ]
            evidence = [item for item in evidence if str(item.school_id).strip() in allowed_ids]

    return hits, evidence, top_schools, ranked_schools, rejected_school, detected_city


def recommend_schools(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
    chat_history: list[dict[str, str]] | None = None,
    user_id: str = "",
    query_understanding_provider: QueryUnderstandingProvider | None = None,
    career_profile: dict[str, Any] | None = None,
) -> RecommendationResult:
    user_question = " ".join(str(question or "").split())
    query_for_context = profile_to_retrieval_query(profile)
    is_profile_placeholder = is_placeholder_recommendation_request(user_question)
    base_question = query_for_context if is_profile_placeholder else (user_question or query_for_context)

    query_understanding: dict[str, Any] = {}
    if user_question and not is_profile_placeholder and query_understanding_provider is not None:
        try:
            query_understanding = query_understanding_provider(
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

    resolved_career_profile = career_profile
    if resolved_career_profile is None and user_id:
        try:
            resolved_career_profile = fetch_user_career_profile(user_id)
        except Exception:
            resolved_career_profile = None

    hits = retrieve(
        question=retrieval_question,
        profile=effective_profile,
        schools=schools,
        transcripts=transcripts,
        top_k=top_k,
        career_profile=resolved_career_profile,
    )

    evidence = _build_evidence(hits)
    top_schools, ranked_schools = _school_rank_payloads(hits, top_k=top_k)
    hits, evidence = _align_hits_and_evidence_to_rank(
        hits=hits,
        evidence=evidence,
        ranked_schools=ranked_schools,
    )
    hits, evidence, top_schools, ranked_schools, rejected_school, detected_city = _apply_context_filters(
        question=user_question,
        chat_history=chat_history,
        hits=hits,
        evidence=evidence,
        top_schools=top_schools,
        ranked_schools=ranked_schools,
    )
    generation_evidence = _select_generation_evidence(evidence, max_items=3)

    return RecommendationResult(
        query_for_context=query_for_context,
        retrieval_question=retrieval_question,
        effective_profile=effective_profile,
        hits=hits,
        evidence=evidence,
        generation_evidence=generation_evidence,
        top_schools=top_schools,
        ranked_schools=ranked_schools,
        rejected_school=rejected_school,
        detected_city=detected_city,
        query_understanding=query_understanding,
        career_profile=resolved_career_profile,
    )
