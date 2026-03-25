from __future__ import annotations

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
            short_answer=(
                "I do not have enough matching data for your current profile and constraints."
            ),
            why_it_fits=(
                "Your budget/country filters removed all candidates. Try widening budget or changing city/country."
            ),
            evidence=[],
            alternative="Consider public options first and ask for computer science or business specifically.",
            next_action="Tell me your target program and whether you can stretch your budget.",
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

    generated = QWEN_GENERATOR.generate(question=question, profile=profile, hits=hits)

    confidence = max(0.2, min(0.95, mean(item.score for item in evidence)))

    return QueryResponse(
        short_answer=generated["short_answer"],
        why_it_fits=generated["why_it_fits"],
        evidence=evidence,
        alternative=generated["alternative"],
        next_action=generated["next_action"],
        confidence=round(confidence, 3),
    )
