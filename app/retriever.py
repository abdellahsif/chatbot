from __future__ import annotations

from dataclasses import dataclass
import re
from threading import Lock
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from app.models import UserProfile

BUDGET_MAX = {
    "zero_public": 0,
    "tight_25k": 25000,
    "comfort_50k": 50000,
    "no_limit_70k_plus": 10**9,
}


@dataclass
class _Entry:
    chunk: dict[str, Any]
    school: dict[str, Any]
    text: str


class _SemanticIndex:
    def __init__(self) -> None:
        self._lock = Lock()
        self._signature: tuple[int, int, str, str] | None = None
        self._entries: list[_Entry] = []
        self._embeddings: torch.Tensor | None = None
        self._embedder: SentenceTransformer | None = None

    def _current_signature(self, schools: dict[str, dict], transcripts: list[dict]) -> tuple[int, int, str, str]:
        first_school = next(iter(schools.keys()), "")
        first_chunk = transcripts[0].get("chunk_id", "") if transcripts else ""
        return (len(schools), len(transcripts), str(first_school), str(first_chunk))

    def _load_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer("intfloat/multilingual-e5-base")
        return self._embedder

    def ensure(self, schools: dict[str, dict], transcripts: list[dict]) -> None:
        signature = self._current_signature(schools, transcripts)
        if self._signature == signature and self._embeddings is not None and self._entries:
            return

        with self._lock:
            if self._signature == signature and self._embeddings is not None and self._entries:
                return

            entries: list[_Entry] = []
            texts: list[str] = []

            for chunk in transcripts:
                school_id = chunk.get("school_id")
                if not school_id:
                    continue
                school = schools.get(school_id)
                if not school:
                    continue

                text = " ".join(
                    [
                        str(chunk.get("text", "")),
                        str(chunk.get("program", "")),
                        str(school.get("name", "")),
                        " ".join(school.get("programs", [])),
                    ]
                )
                entries.append(_Entry(chunk=chunk, school=school, text=text))
                texts.append(f"passage: {text}")

            if not entries:
                self._entries = []
                self._embeddings = None
                self._signature = signature
                return

            embedder = self._load_embedder()
            embeddings = embedder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )
            self._entries = entries
            self._embeddings = embeddings.cpu()
            self._signature = signature

    def query(
        self,
        question: str,
        profile: UserProfile,
        top_k: int,
    ) -> list[dict]:
        if self._embeddings is None or not self._entries:
            return []

        valid_indices: list[int] = []
        for i, entry in enumerate(self._entries):
            if school_matches_profile(entry.school, profile):
                valid_indices.append(i)

        if not valid_indices:
            return []

        embedder = self._load_embedder()
        q = embedder.encode(
            [f"query: {question}"],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].cpu()

        index_tensor = torch.tensor(valid_indices, dtype=torch.long)
        sub_embeddings = self._embeddings.index_select(0, index_tensor)
        scores = torch.matmul(sub_embeddings, q)

        shortlist = min(max(top_k * 6, 20), len(valid_indices))
        top_scores, top_positions = torch.topk(scores, k=shortlist)

        candidates: list[dict] = []
        for score, pos in zip(top_scores.tolist(), top_positions.tolist()):
            global_index = valid_indices[pos]
            entry = self._entries[global_index]
            candidates.append({"score": float(score), "chunk": entry.chunk, "school": entry.school})

        return candidates


SEMANTIC_INDEX = _SemanticIndex()


def budget_allows(profile_budget: str, tuition_max_mad: int) -> bool:
    if not isinstance(tuition_max_mad, int):
        return False
    if profile_budget == "zero_public":
        return tuition_max_mad <= 12000
    cap = BUDGET_MAX.get(profile_budget)
    if cap is None:
        return tuition_max_mad <= BUDGET_MAX["comfort_50k"]
    return tuition_max_mad <= cap


def school_matches_profile(school: dict, profile: UserProfile) -> bool:
    if str(school.get("country", "")).upper() != profile.country:
        return False
    tuition_max = school.get("tuition_max_mad")
    if not isinstance(tuition_max, int):
        try:
            tuition_max = int(tuition_max)
        except (TypeError, ValueError):
            return False
    if not budget_allows(profile.budget_band, tuition_max):
        return False
    return True


def retrieve(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> list[dict]:
    SEMANTIC_INDEX.ensure(schools, transcripts)
    candidates = SEMANTIC_INDEX.query(question=question, profile=profile, top_k=top_k)
    if not candidates:
        return []

    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    query_tokens = _tokenize(question)
    rescored: list[dict] = []

    for item in candidates:
        school = item["school"]
        chunk = item["chunk"]

        semantic_score = float(item["score"])
        lexical_tokens = _tokenize(
            " ".join(
                [
                    str(chunk.get("program", "")),
                    str(chunk.get("text", "")),
                    str(school.get("name", "")),
                    " ".join(school.get("programs", [])),
                ]
            )
        )
        lexical_score = (len(query_tokens & lexical_tokens) / len(query_tokens)) if query_tokens else 0.0

        profile_bonus = 0.0
        school_city = str(school.get("city", "")).strip().lower()
        if profile.city and school_city == profile.city.strip().lower():
            profile_bonus += 0.03

        program_tokens = _tokenize(str(chunk.get("program", "")))
        if program_tokens and (query_tokens & program_tokens):
            profile_bonus += 0.02

        final_score = (0.75 * semantic_score) + (0.20 * lexical_score) + profile_bonus
        rescored.append(
            {
                "score": float(final_score),
                "semantic_score": float(semantic_score),
                "lexical_score": float(lexical_score),
                "chunk": chunk,
                "school": school,
            }
        )

    rescored.sort(key=lambda x: x["score"], reverse=True)

    school_cap = 2 if top_k >= 3 else top_k
    selected: list[dict] = []
    per_school_counts: dict[str, int] = {}

    for item in rescored:
        school = item["school"]
        school_key = str(school.get("school_id") or school.get("name") or "")
        count = per_school_counts.get(school_key, 0)
        if count >= school_cap:
            continue
        selected.append(item)
        per_school_counts[school_key] = count + 1
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in rescored:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    return selected
