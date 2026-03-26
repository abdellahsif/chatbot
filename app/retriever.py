from __future__ import annotations

from dataclasses import dataclass
import re
from threading import Lock
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

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


@dataclass
class _SchoolDoc:
    school_id: str
    school: dict[str, Any]
    text: str
    chunks: list[dict[str, Any]]


class _SemanticIndex:
    def __init__(self) -> None:
        self._lock = Lock()
        self._signature: tuple[int, int, str, str] | None = None
        self._entries: list[_Entry] = []
        self._school_docs: list[_SchoolDoc] = []
        self._embeddings: torch.Tensor | None = None
        self._school_embeddings: np.ndarray | None = None
        self._faiss_index: Any = None
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
            chunk_by_school: dict[str, list[dict[str, Any]]] = {}

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
                chunk_by_school.setdefault(str(school_id), []).append(chunk)

            if not entries:
                self._entries = []
                self._school_docs = []
                self._embeddings = None
                self._school_embeddings = None
                self._faiss_index = None
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

            school_docs: list[_SchoolDoc] = []
            school_texts: list[str] = []
            for school_id, school in schools.items():
                chunks = chunk_by_school.get(str(school_id), [])
                chunk_text = " ".join(str(c.get("text", ""))[:220] for c in chunks[:3])
                doc_text = " ".join(
                    [
                        str(school.get("name", "")),
                        str(school.get("city", "")),
                        " ".join(school.get("programs", [])),
                        str(school.get("type", "")),
                        str(school.get("tuition_max_mad", "")),
                        chunk_text,
                    ]
                )
                school_docs.append(
                    _SchoolDoc(
                        school_id=str(school_id),
                        school=school,
                        text=doc_text,
                        chunks=chunks,
                    )
                )
                school_texts.append(f"passage: {doc_text}")

            school_embeddings = embedder.encode(
                school_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            ).astype("float32")

            faiss_index = None
            if faiss is not None and len(school_embeddings) > 0:
                faiss_index = faiss.IndexFlatIP(int(school_embeddings.shape[1]))
                faiss_index.add(school_embeddings)

            self._entries = entries
            self._school_docs = school_docs
            self._embeddings = embeddings.cpu()
            self._school_embeddings = school_embeddings
            self._faiss_index = faiss_index
            self._signature = signature

    def query_schools(
        self,
        question: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self._school_docs:
            return []

        embedder = self._load_embedder()
        q_np = embedder.encode(
            [f"query: {question}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")

        retrieve_k = min(max(top_k, 5), 10)
        retrieve_k = min(retrieve_k, len(self._school_docs))

        candidates: list[dict[str, Any]] = []
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(np.expand_dims(q_np, axis=0), retrieve_k)
            for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
                if idx < 0:
                    continue
                doc = self._school_docs[idx]
                candidates.append(
                    {
                        "semantic_score": float(score),
                        "school": doc.school,
                        "school_id": doc.school_id,
                        "chunks": doc.chunks,
                        "text": doc.text,
                    }
                )
        elif self._school_embeddings is not None:
            scores = np.dot(self._school_embeddings, q_np)
            best = np.argsort(-scores)[:retrieve_k]
            for idx in best.tolist():
                doc = self._school_docs[idx]
                candidates.append(
                    {
                        "semantic_score": float(scores[idx]),
                        "school": doc.school,
                        "school_id": doc.school_id,
                        "chunks": doc.chunks,
                        "text": doc.text,
                    }
                )

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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


INTENT_SYNONYMS: dict[str, set[str]] = {
    "data": {"data", "analyst", "analytics", "bi", "business", "intelligence", "ai", "ml", "stat", "statistique"},
    "software": {"software", "dev", "developer", "full", "stack", "web", "mobile"},
    "cyber": {"cyber", "security", "securite", "soc", "forensics"},
}


def _expanded_query_tokens(question: str) -> set[str]:
    tokens = _tokenize(question)
    expanded = set(tokens)
    for group in INTENT_SYNONYMS.values():
        if tokens & group:
            expanded |= group
    return expanded


def _has_explicit_program_intent(question: str) -> bool:
    q = _tokenize(question)
    return any(q & group for group in INTENT_SYNONYMS.values())


def _expected_grade_to_level(expected_grade_band: str) -> float:
    text = (expected_grade_band or "").strip().lower()
    mapping = {
        "passable": 0.35,
        "bien": 0.55,
        "tres_bien": 0.78,
        "elite": 0.95,
        "10_12": 0.45,
        "12_14": 0.55,
        "14_16": 0.75,
        "16_20": 0.95,
    }
    return mapping.get(text, 0.6)


def _selectivity_to_required_level(selectivity: str) -> float:
    text = (selectivity or "").strip().lower()
    mapping = {"low": 0.35, "medium": 0.55, "high": 0.8}
    return mapping.get(text, 0.55)


def _bac_stream_compatible(profile_bac_stream: str, chunks: list[dict[str, Any]], school: dict[str, Any]) -> bool:
    bac = (profile_bac_stream or "").strip().lower()
    if not bac:
        return True
    known = {
        "sm": ["sm", "science_math", "science math"],
        "science_math": ["sm", "science_math", "science math"],
        "spc": ["spc", "pc", "physique"],
        "svt": ["svt", "bio"],
        "eco": ["eco", "econom"],
        "lettres": ["lettres", "literature"],
    }
    aliases = known.get(bac, [bac])
    searchable = " ".join(str(c.get("text", "")).lower() for c in chunks[:5])
    searchable += " " + " ".join(str(p).lower() for p in school.get("programs", []))

    # If dataset does not specify required series, keep candidate.
    if "serie bac" not in searchable and "bac" not in searchable:
        return True
    return any(alias in searchable for alias in aliases)


def _budget_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    tuition_max = school.get("tuition_max_mad")
    if not isinstance(tuition_max, int):
        return 0.0
    if budget_allows(profile.budget_band, tuition_max):
        return 1.0
    return 0.0


def _program_match_score(question: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    q_tokens = _expanded_query_tokens(question)
    if not q_tokens:
        return 0.0
    school_text = " ".join(school.get("programs", []))
    chunk_text = " ".join(str(c.get("program", "")) for c in chunks[:5])
    target_tokens = _tokenize(f"{school_text} {chunk_text}")
    overlap = len(q_tokens & target_tokens)
    return min(1.0, overlap / max(1, len(q_tokens)))


def _grade_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    expected = _expected_grade_to_level(profile.expected_grade_band)
    required = _selectivity_to_required_level(str(school.get("admission_selectivity", "medium")))
    diff = abs(expected - required)
    return max(0.0, 1.0 - diff)


def _location_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    if profile.city and str(school.get("city", "")).strip().lower() == profile.city.strip().lower():
        return 1.0
    if str(school.get("country", "")).upper() == profile.country:
        return 0.4
    return 0.0


def _score_candidate(question: str, profile: UserProfile, school: dict[str, Any], chunks: list[dict[str, Any]], semantic: float) -> dict[str, float]:
    program_match = _program_match_score(question, school, chunks)
    budget_match = _budget_match_score(profile, school)
    grade_match = _grade_match_score(profile, school)
    location_match = _location_match_score(profile, school)

    weighted = (
        0.4 * program_match
        + 0.3 * budget_match
        + 0.2 * grade_match
        + 0.1 * location_match
    )
    final_score = 0.8 * weighted + 0.2 * max(0.0, semantic)
    return {
        "program_match": program_match,
        "budget_match": budget_match,
        "grade_match": grade_match,
        "location_match": location_match,
        "weighted": weighted,
        "final": final_score,
    }


def retrieve(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> list[dict]:
    SEMANTIC_INDEX.ensure(schools, transcripts)
    candidates = SEMANTIC_INDEX.query_schools(question=question, top_k=max(5, min(10, top_k)))
    if not candidates:
        return []

    filtered: list[dict] = []
    for item in candidates:
        school = item["school"]
        chunks = item.get("chunks", [])
        if not school_matches_profile(school, profile):
            continue
        if not _bac_stream_compatible(profile.bac_stream, chunks, school):
            continue
        filtered.append(item)

    if not filtered:
        return []

    rescored: list[dict] = []
    strict_intent = _has_explicit_program_intent(question)
    for item in filtered:
        school = item["school"]
        chunks = item.get("chunks", [])
        semantic = float(item.get("semantic_score", 0.0))
        components = _score_candidate(question, profile, school, chunks, semantic)

        # If user clearly asks for a domain (e.g., data analyst), drop weak program matches.
        if strict_intent and components["program_match"] < 0.10:
            continue

        evidence_chunks = sorted(
            chunks,
            key=lambda c: _program_match_score(question, school, [c]),
            reverse=True,
        )
        chosen_chunk = evidence_chunks[0] if evidence_chunks else {
            "chunk_id": f"school_{school.get('school_id', school.get('name', 'unknown'))}",
            "school_id": school.get("school_id", ""),
            "program": (school.get("programs", ["general"]) or ["general"])[0],
            "recorded_at": "2026-01-01",
            "text": item.get("text", ""),
        }

        rescored.append(
            {
                "score": float(components["final"]),
                "chunk": chosen_chunk,
                "school": school,
                "score_components": components,
            }
        )

    rescored.sort(key=lambda x: x["score"], reverse=True)
    if not rescored:
        return []
    select_n = min(max(3, top_k), 5)
    return rescored[:select_n]
