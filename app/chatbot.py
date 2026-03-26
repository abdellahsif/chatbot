from __future__ import annotations

import re
from statistics import mean

from app.generator import QWEN_GENERATOR
from app.models import EvidenceItem, QueryResponse, UserProfile
from app.retriever import retrieve


def answer_question(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> QueryResponse:
    hits = retrieve(
        question=question,
        profile=profile,
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

    generated = QWEN_GENERATOR.generate(
        question=question,
        profile=profile,
        top_schools=top_schools,
    )

    question_hint = " ".join(re.findall(r"[a-z0-9]+", question.lower())[:8])
    evidence_hint = " ".join(re.findall(r"[a-z0-9]+", evidence[0].text.lower())[:20]) if evidence else ""

    short_answer = generated["short_answer"]
    why_it_fits = generated["why_it_fits"]

    if question_hint and question_hint not in short_answer.lower():
        short_answer = f"{short_answer} Question focus: {question_hint}."
    if evidence_hint and "evidence snippet" not in why_it_fits.lower():
        why_it_fits = f"{why_it_fits} Evidence snippet: {evidence_hint}."

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))

    return QueryResponse(
        short_answer=short_answer,
        why_it_fits=why_it_fits,
        evidence=evidence,
        alternative=generated["alternative"],
        next_action=generated["next_action"],
        confidence=round(confidence, 3),
    )
