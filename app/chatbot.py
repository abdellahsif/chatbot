from __future__ import annotations

from statistics import mean

from app.models import EvidenceItem, QueryResponse, UserProfile
from app.retriever import resolve_effective_profile, retrieve


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
    short_answer = f"Best match: {top_ev.school_name}."

    ev_text = " ".join(str(top_ev.text).split())
    ev_excerpt = " ".join(ev_text.split()[:28])
    why_it_fits = f"Evidence snippet: {ev_excerpt}."

    if len(evidence) > 1:
        alt_school = evidence[1].school_name
        alt_text = " ".join(str(evidence[1].text).split())
        alt_excerpt = " ".join(alt_text.split()[:20])
    else:
        alt_school = top_ev.school_name
        alt_excerpt = ev_excerpt

    alternative = f"Alternative {alt_school}: {alt_excerpt}."
    next_action = f"Evidence focus: {ev_excerpt}."

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))

    return QueryResponse(
        short_answer=short_answer,
        why_it_fits=why_it_fits,
        evidence=evidence,
        alternative=alternative,
        next_action=next_action,
        confidence=round(confidence, 3),
    )
