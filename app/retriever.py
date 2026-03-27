from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
import re
from threading import Lock
from typing import Any
import unicodedata

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
                dim = int(school_embeddings.shape[1])
                use_hnsw = _env_bool("USE_HNSW_INDEX", True)
                if use_hnsw and hasattr(faiss, "IndexHNSWFlat"):
                    hnsw_m = max(8, _env_int("HNSW_M", 32))
                    metric = getattr(faiss, "METRIC_INNER_PRODUCT", 0)
                    try:
                        hnsw = faiss.IndexHNSWFlat(dim, hnsw_m, metric)
                    except TypeError:
                        hnsw = faiss.IndexHNSWFlat(dim, hnsw_m)
                    hnsw.hnsw.efConstruction = max(20, _env_int("HNSW_EF_CONSTRUCTION", 200))
                    hnsw.hnsw.efSearch = max(20, _env_int("HNSW_EF_SEARCH", 128))
                    faiss_index = hnsw
                else:
                    faiss_index = faiss.IndexFlatIP(dim)
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
        retrieve_k = min(max(top_k, 5), max(10, _env_int("DENSE_RETRIEVE_MAX", 60)))
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


class _SparseSchoolIndex:
    def __init__(self) -> None:
        self._lock = Lock()
        self._signature: tuple[int, int, str, str] | None = None
        self._docs: list[_SchoolDoc] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None

    def _current_signature(self, schools: dict[str, dict], transcripts: list[dict]) -> tuple[int, int, str, str]:
        first_school = next(iter(schools.keys()), "")
        first_chunk = transcripts[0].get("chunk_id", "") if transcripts else ""
        return (len(schools), len(transcripts), str(first_school), str(first_chunk))

    def ensure(self, schools: dict[str, dict], transcripts: list[dict]) -> None:
        signature = self._current_signature(schools, transcripts)
        if self._signature == signature and self._vectorizer is not None and self._matrix is not None:
            return

        with self._lock:
            if self._signature == signature and self._vectorizer is not None and self._matrix is not None:
                return

            chunk_by_school: dict[str, list[dict[str, Any]]] = {}
            for chunk in transcripts:
                school_id = str(chunk.get("school_id", "")).strip()
                if school_id:
                    chunk_by_school.setdefault(school_id, []).append(chunk)

            docs: list[_SchoolDoc] = []
            texts: list[str] = []
            for school_id, school in schools.items():
                sid = str(school_id)
                chunks = chunk_by_school.get(sid, [])
                chunk_text = " ".join(str(c.get("text", "")) for c in chunks[:6])
                doc_text = " ".join(
                    [
                        str(school.get("name", "")),
                        str(school.get("city", "")),
                        " ".join(school.get("programs", [])),
                        str(school.get("type", "")),
                        chunk_text,
                    ]
                )
                docs.append(_SchoolDoc(school_id=sid, school=school, text=doc_text, chunks=chunks))
                texts.append(doc_text)

            if not texts:
                self._docs = []
                self._vectorizer = None
                self._matrix = None
                self._signature = signature
                return

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform(texts)

            self._docs = docs
            self._vectorizer = vectorizer
            self._matrix = matrix
            self._signature = signature

    def query_schools(self, question: str, top_k: int) -> list[dict[str, Any]]:
        if not self._docs or self._vectorizer is None or self._matrix is None:
            return []

        retrieve_k = min(max(top_k, 5), max(10, _env_int("SPARSE_RETRIEVE_MAX", 80)))
        retrieve_k = min(retrieve_k, len(self._docs))
        q_vec = self._vectorizer.transform([question])
        scores = linear_kernel(q_vec, self._matrix)[0]
        best = np.argsort(-scores)[:retrieve_k]

        out: list[dict[str, Any]] = []
        for idx in best.tolist():
            doc = self._docs[idx]
            out.append(
                {
                    "sparse_score": float(scores[idx]),
                    "school": doc.school,
                    "school_id": doc.school_id,
                    "chunks": doc.chunks,
                    "text": doc.text,
                }
            )
        return out


SEMANTIC_INDEX = _SemanticIndex()
SPARSE_INDEX = _SparseSchoolIndex()


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
    return True


def _tokenize(text: str) -> set[str]:
    lowered = (text or "").lower()
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    return set(re.findall(r"[a-z0-9]+", folded))


INTENT_SYNONYMS: dict[str, set[str]] = {
    "data": {"data", "analyst", "analytics", "bi", "intelligence", "ai", "ml", "stat", "statistique", "science", "informatique"},
    "software": {"software", "dev", "developer", "full", "stack", "web", "mobile", "it", "programming", "code", "developpement", "cs", "computer", "computing", "informatics", "informatique"},
    "cyber": {"cyber", "security", "securite", "soc", "forensics", "infosec"},
    "business": {"business", "commerce", "management", "gestion", "finance", "marketing", "entreprise"},
    "architecture": {"architecture", "urban", "urbanisme", "design", "amenagement"},
    "health": {"health", "healthcare", "sante", "paramedical", "medical", "nursing", "care"},
    "arts": {"art", "arts", "beaux", "design", "portfolio", "creative", "cinema"},
    "military": {"military", "armee", "defense", "officier", "royale"},
    "vocational": {"ofppt", "technician", "technicien", "technologique", "pratique", "credential"},
}

DOMAIN_TERMS: dict[str, set[str]] = {
    "law": {"law", "droit", "juridique", "legal", "jurisprudence"},
    "medicine": {"medicine", "medical", "medecine", "pharmacie"},
    "healthcare": {"health", "healthcare", "sante", "paramedical", "nursing", "care", "sanitaire"},
    "engineering": {"engineering", "ingenieur", "ingenierie", "technologie", "tech"},
    "business": {"business", "commerce", "gestion", "finance", "management", "economie"},
    "arts": {"art", "arts", "design", "beaux", "cinema"},
}

LANGUAGE_TERMS: dict[str, set[str]] = {
    "arabic": {"arabic", "arabe"},
    "french": {"french", "francais", "fran", "français"},
    "english": {"english", "anglais"},
    "bilingual": {"bilingual", "bilingue", "multilingual", "multilingue"},
}

INSTITUTION_TYPE_TERMS: dict[str, set[str]] = {
    "faculty": {"faculty", "faculte", "facultes", "facult"},
    "institute": {"institute", "institut", "institute"},
    "school": {"school", "ecole", "école"},
    "university": {"university", "universite", "universit"},
}

def _expanded_query_tokens(question: str) -> set[str]:
    tokens = _tokenize(question)
    expanded = set(tokens)
    for group in INTENT_SYNONYMS.values():
        if tokens & group:
            expanded |= group
    return expanded


def _extract_query_constraints(question: str) -> dict[str, Any]:
    q_tokens = _tokenize(question)
    q_norm = _normalize_city_text(question)

    domains = {name for name, terms in DOMAIN_TERMS.items() if q_tokens & terms}
    languages = {name for name, terms in LANGUAGE_TERMS.items() if q_tokens & _tokenize(" ".join(terms))}
    institution_types = {
        name
        for name, terms in INSTITUTION_TYPE_TERMS.items()
        if q_tokens & _tokenize(" ".join(terms))
    }

    language_mode = "all"
    if languages and re.search(r"\b(or|ou)\b", q_norm):
        language_mode = "any"

    neg_markers = {"not", "without", "except", "pas", "hors"}
    q_words = q_norm.split()
    excluded_domains: set[str] = set()
    for domain, terms in DOMAIN_TERMS.items():
        for term in terms:
            term_words = term.split()
            if not term_words:
                continue
            n = len(term_words)
            for i in range(0, max(0, len(q_words) - n + 1)):
                if q_words[i : i + n] != term_words:
                    continue
                window = set(q_words[max(0, i - 3) : i])
                if window & neg_markers:
                    excluded_domains.add(domain)
                    break
            if domain in excluded_domains:
                break

    domains = {d for d in domains if d not in excluded_domains}

    return {
        "domains": domains,
        "excluded_domains": excluded_domains,
        "languages": languages,
        "institution_types": institution_types,
        "language_mode": language_mode,
        "has_constraints": bool(domains or excluded_domains or languages or institution_types),
    }


def _school_domain_tokens(school: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    text = " ".join(
        [
            str(school.get("name", "")),
            str(school.get("type", "")),
            " ".join(school.get("programs", [])),
            " ".join(str(c.get("program", "")) for c in chunks[:6]),
            " ".join(str(c.get("text", "")) for c in chunks[:3]),
        ]
    )
    return _tokenize(text)


def _school_language_tags(school: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    text = " ".join(
        [
            str(school.get("name", "")),
            " ".join(str(c.get("language", "")) for c in chunks[:6]),
            " ".join(str(c.get("text", "")) for c in chunks[:3]),
        ]
    )
    toks = _tokenize(text)
    langs: set[str] = set()
    if toks & {"arabe", "arabic"}:
        langs.add("arabic")
    if toks & {"francais", "french"}:
        langs.add("french")
    if toks & {"anglais", "english"}:
        langs.add("english")
    if len(langs) >= 2:
        langs.add("bilingual")
    return langs


def _school_institution_type_tags(school: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    text_tokens = _school_domain_tokens(school, chunks)
    tags: set[str] = set()
    if text_tokens & {"faculte", "faculty"}:
        tags.add("faculty")
    if text_tokens & {"institut", "institute"}:
        tags.add("institute")
    if text_tokens & {"ecole", "school"}:
        tags.add("school")
    if text_tokens & {"universite", "university"}:
        tags.add("university")
    return tags


def _school_matches_query_constraints(
    school: dict[str, Any],
    chunks: list[dict[str, Any]],
    constraints: dict[str, Any],
) -> bool:
    if not constraints.get("has_constraints", False):
        return True

    domain_tokens = _school_domain_tokens(school, chunks)

    domains: set[str] = constraints.get("domains", set())
    if domains:
        domain_ok = False
        for domain in domains:
            terms = DOMAIN_TERMS.get(domain, set())
            if domain_tokens & terms:
                domain_ok = True
                break
        if not domain_ok:
            return False

    excluded_domains: set[str] = constraints.get("excluded_domains", set())
    if excluded_domains:
        for domain in excluded_domains:
            terms = DOMAIN_TERMS.get(domain, set())
            if domain_tokens & terms:
                return False

    languages: set[str] = constraints.get("languages", set())
    if languages:
        school_langs = _school_language_tags(school, chunks)
        mode = constraints.get("language_mode", "all")
        if mode == "any":
            if not (school_langs & languages):
                return False
        else:
            if not languages.issubset(school_langs):
                return False

    institution_types: set[str] = constraints.get("institution_types", set())
    if institution_types:
        school_types = _school_institution_type_tags(school, chunks)
        if not (school_types & institution_types):
            return False

    return True


def _normalize_city_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"[^a-z0-9\s/,-]+", " ", folded)
    return " ".join(folded.split())


def _extract_city_intent(question: str, schools: dict[str, dict]) -> str | None:
    q = _normalize_city_text(question)
    if not q:
        return None

    known_cities: set[str] = set()
    for school in schools.values():
        city = _normalize_city_text(str(school.get("city", "")))
        if not city:
            continue
        # Split multi-campus city fields like "Casablanca / Rabat / ...".
        for part in re.split(r"[/,;-]+", city):
            part = " ".join(part.split())
            if part:
                known_cities.add(part)

    if not known_cities:
        return None

    # Exact / substring match first.
    for city in sorted(known_cities, key=len, reverse=True):
        if city in q:
            return city

    q_tokens = q.split()
    has_location_cue = bool(
        re.search(
            r"\b(in|at|near|from|to|city|ville|en|au|aux)\b",
            q,
        )
    )
    # Avoid accidental fuzzy city matches on long non-location questions.
    if not has_location_cue and len(q_tokens) > 3:
        return None

    # Fuzzy match on question tokens and bi-grams for typo tolerance.
    q_phrases: set[str] = set(q_tokens)
    for i in range(len(q_tokens) - 1):
        q_phrases.add(f"{q_tokens[i]} {q_tokens[i + 1]}")

    best_city = None
    best_score = 0.0
    for phrase in q_phrases:
        for city in known_cities:
            score = difflib.SequenceMatcher(a=phrase, b=city).ratio()
            if score > best_score:
                best_score = score
                best_city = city

    if best_city is not None and best_score >= 0.72:
        return best_city
    return None


def _city_matches_intent(school_city: str, city_intent: str | None) -> bool:
    if not city_intent:
        return True
    city_text = _normalize_city_text(school_city)
    if not city_text:
        return False

    # Handle multi-campus city fields like "Casablanca / Rabat / ...".
    parts = [p.strip() for p in re.split(r"[/,;-]+", city_text) if p.strip()]
    candidates = set(parts + [city_text])

    for candidate in candidates:
        if candidate == city_intent:
            return True
        if city_intent in candidate or candidate in city_intent:
            return True
        if difflib.SequenceMatcher(a=candidate, b=city_intent).ratio() >= 0.84:
            return True
    return False


def _school_breadth_score(school: dict[str, Any]) -> float:
    programs = school.get("programs", [])
    count = len(programs) if isinstance(programs, list) else 0
    if count <= 0:
        return 0.0
    return min(1.0, count / 10.0)


def _is_location_only_query(question: str, city_intent: str | None) -> bool:
    if not city_intent:
        return False
    q_tokens = _tokenize(question)
    if not q_tokens:
        return False

    generic = {
        "i",
        "want",
        "to",
        "study",
        "in",
        "at",
        "the",
        "a",
        "an",
        "me",
        "my",
        "for",
        "school",
        "university",
        "universite",
    }
    city_tokens = set(_normalize_city_text(city_intent).split())

    def _looks_like_city_token(token: str) -> bool:
        if token in city_tokens:
            return True
        if len(token) < 4:
            return False
        return any(difflib.SequenceMatcher(a=token, b=ct).ratio() >= 0.72 for ct in city_tokens)

    core = {t for t in q_tokens if t not in generic and not _looks_like_city_token(t)}
    return len(core) == 0 and not _has_explicit_program_intent(question)


def _build_query_variants(question: str, profile: UserProfile) -> list[str]:
    base = (question or "").strip()
    if not base:
        return []

    expanded_tokens = sorted(_expanded_query_tokens(base))
    variants = [base]

    if expanded_tokens:
        variants.append(" ".join(expanded_tokens))

    profile_bits = [
        (profile.bac_stream or "").strip(),
        (profile.motivation or "").strip(),
        (profile.city or "").strip(),
    ]
    profile_hint = " ".join(b for b in profile_bits if b)
    if profile_hint:
        variants.append(f"{base} {profile_hint}".strip())

    dedup: list[str] = []
    seen: set[str] = set()
    for v in variants:
        key = " ".join(v.lower().split())
        if key and key not in seen:
            seen.add(key)
            dedup.append(v)
    return dedup[:4]


def _has_explicit_program_intent(question: str) -> bool:
    q = _tokenize(question)
    return any(q & group for group in INTENT_SYNONYMS.values())


def _acronym_from_name(name: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]+", name or "") if len(w) >= 2]
    if len(words) < 2:
        return ""
    return "".join(w[0].lower() for w in words)


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


def _bac_stream_match_score(profile_bac_stream: str, chunks: list[dict[str, Any]], school: dict[str, Any]) -> float:
    return 1.0 if _bac_stream_compatible(profile_bac_stream, chunks, school) else 0.4


def _intent_group_match_score(question: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0

    matched_groups: list[set[str]] = []
    for group in INTENT_SYNONYMS.values():
        if q_tokens & group:
            matched_groups.append(group)

    if not matched_groups:
        return 0.0

    school_text = " ".join(
        [
            str(school.get("name", "")),
            str(school.get("type", "")),
            " ".join(school.get("programs", [])),
            " ".join(str(c.get("program", "")) for c in chunks[:6]),
            " ".join(str(c.get("text", "")) for c in chunks[:4]),
        ]
    )
    target_tokens = _tokenize(school_text)
    if not target_tokens:
        return 0.0

    best = 0.0
    for group in matched_groups:
        overlap = len(group & target_tokens)
        score = _safe_div(overlap, max(1, len(group)))
        best = max(best, score)
    return min(1.0, best * 2.0)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _budget_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    tuition_max = _to_int(school.get("tuition_max_mad"), default=10**9)
    if budget_allows(profile.budget_band, tuition_max):
        return 1.0
    band_cap = BUDGET_MAX.get(profile.budget_band, BUDGET_MAX["comfort_50k"])
    if band_cap <= 0:
        return 0.1
    over_ratio = max(0.0, (tuition_max - band_cap) / float(max(1, band_cap)))
    if over_ratio <= 0.25:
        return 0.65
    if over_ratio <= 0.6:
        return 0.35
    return 0.1


def _program_match_score(question: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    q_tokens = _expanded_query_tokens(question)
    intent_tokens = set().union(*INTENT_SYNONYMS.values())
    focused_q = q_tokens & intent_tokens
    strict_intent = _has_explicit_program_intent(question)
    if focused_q and strict_intent:
        q_tokens = focused_q
    if not q_tokens:
        return 0.0
    school_text = " ".join(school.get("programs", []))
    chunk_text = " ".join(str(c.get("program", "")) for c in chunks[:5])
    target_tokens = _tokenize(f"{school_text} {chunk_text}")
    overlap = len(q_tokens & target_tokens)
    base = min(1.0, overlap / max(1, len(q_tokens)))

    # Soft fallback: reward near matches when explicit terms are short or variant.
    q_text = " ".join(sorted(q_tokens))
    t_text = " ".join(sorted(target_tokens))
    soft = 0.0
    if q_text and t_text:
        hits = sum(1 for tok in q_tokens if len(tok) >= 4 and tok in t_text)
        soft = min(1.0, hits / max(1, len(q_tokens)))
    return max(base, 0.7 * base + 0.3 * soft)


def _grade_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    expected = _expected_grade_to_level(profile.expected_grade_band)
    required = _selectivity_to_required_level(str(school.get("admission_selectivity", "medium")))
    diff = abs(expected - required)
    return max(0.0, 1.0 - diff)


def _location_match_score(profile: UserProfile, school: dict[str, Any], city_intent: str | None = None) -> float:
    school_city = str(school.get("city", "")).strip()
    if city_intent:
        return 1.0 if _city_matches_intent(school_city, city_intent) else 0.0
    if profile.city and school_city.lower() == profile.city.strip().lower():
        return 1.0
    if str(school.get("country", "")).upper() == profile.country:
        return 0.4
    return 0.0


def _motivation_match_score(profile: UserProfile, school: dict[str, Any]) -> float:
    motivation = (profile.motivation or "").strip().lower()
    selectivity = str(school.get("admission_selectivity", "medium")).strip().lower()
    employability = float(school.get("employability_score", 0.0) or 0.0)
    tuition_max = _to_int(school.get("tuition_max_mad"), default=10**9)
    salary_min = _to_int(school.get("salary_entry_min_mad"), default=0)
    salary_max = _to_int(school.get("salary_entry_max_mad"), default=0)
    avg_salary = (salary_min + salary_max) / 2.0 if (salary_min or salary_max) else 0.0
    has_international = str(school.get("international_double_degree", "false")).strip().lower() == "true"

    if motivation == "cash":
        if tuition_max <= 0:
            return 1.0
        roi = avg_salary / float(max(1, tuition_max))
        return min(1.0, max(0.0, roi * 6.0))
    if motivation == "prestige":
        selectivity_bonus = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(selectivity, 0.5)
        return min(1.0, 0.6 * selectivity_bonus + 0.4 * min(1.0, employability / 5.0))
    if motivation == "expat":
        return 1.0 if has_international else 0.25
    if motivation == "safety":
        budget_safety = 1.0 if budget_allows(profile.budget_band, tuition_max) else 0.3
        selectivity_safety = {"low": 1.0, "medium": 0.75, "high": 0.4}.get(selectivity, 0.6)
        return 0.6 * budget_safety + 0.4 * selectivity_safety
    return 0.5


def _extract_school_mentions(question: str, schools: dict[str, dict]) -> set[str]:
    q = (question or "").lower()
    q_tokens = _tokenize(question)
    mentioned: set[str] = set()

    for school_id, school in schools.items():
        school_id_str = str(school_id)
        name = str(school.get("name", ""))
        name_tokens = _tokenize(name)
        if not name_tokens:
            continue

        acronym = _acronym_from_name(name)
        if acronym and acronym in q:
            mentioned.add(school_id_str)

        short_aliases = {
            token
            for token in name_tokens
            if len(token) >= 4 and not token.isdigit()
        }
        if short_aliases & q_tokens:
            mentioned.add(school_id_str)

        overlap = _safe_div(len(name_tokens & q_tokens), len(name_tokens))
        if overlap >= 0.5:
            mentioned.add(school_id_str)
    return mentioned


def _score_candidate(
    question: str,
    profile: UserProfile,
    school: dict[str, Any],
    chunks: list[dict[str, Any]],
    semantic: float,
    city_intent: str | None = None,
) -> dict[str, float]:
    program_match = _program_match_score(question, school, chunks)
    intent_match = _intent_group_match_score(question, school, chunks)
    bac_match = _bac_stream_match_score(profile.bac_stream, chunks, school)
    budget_match = _budget_match_score(profile, school)
    grade_match = _grade_match_score(profile, school)
    location_match = _location_match_score(profile, school, city_intent=city_intent)
    motivation_match = _motivation_match_score(profile, school)

    weighted = (
        0.2 * program_match
        + 0.2 * intent_match
        + 0.15 * bac_match
        + 0.1 * budget_match
        + 0.1 * grade_match
        + 0.1 * location_match
        + 0.15 * motivation_match
    )
    final_score = 0.45 * weighted + 0.55 * max(0.0, semantic)
    return {
        "program_match": program_match,
        "intent_match": intent_match,
        "bac_match": bac_match,
        "budget_match": budget_match,
        "grade_match": grade_match,
        "location_match": location_match,
        "motivation_match": motivation_match,
        "weighted": weighted,
        "final": final_score,
    }


def _lexical_match_score(question: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    q_tokens = _expanded_query_tokens(question)
    if not q_tokens:
        return 0.0

    school_text = " ".join(
        [
            str(school.get("name", "")),
            str(school.get("city", "")),
            " ".join(school.get("programs", [])),
            str(school.get("type", "")),
        ]
    )
    chunk_text = " ".join(str(c.get("text", "")) for c in chunks[:4])
    target_tokens = _tokenize(f"{school_text} {chunk_text}")
    if not target_tokens:
        return 0.0

    overlap = len(q_tokens & target_tokens)
    return min(1.0, _safe_div(overlap, max(1, min(len(q_tokens), 12))))


def retrieve(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> list[dict]:
    SEMANTIC_INDEX.ensure(schools, transcripts)
    SPARSE_INDEX.ensure(schools, transcripts)
    city_intent = _extract_city_intent(question, schools)
    query_constraints = _extract_query_constraints(question)
    location_only_query = _is_location_only_query(question, city_intent)
    candidate_k = max(40, min(120, top_k * 12))
    variants = _build_query_variants(question, profile)

    merged: dict[str, dict[str, Any]] = {}
    for q in variants or [question]:
        for cand in SEMANTIC_INDEX.query_schools(question=q, top_k=candidate_k):
            sid = str(cand.get("school_id", "")).strip() or str(cand.get("school", {}).get("school_id", "")).strip()
            if not sid:
                continue
            prev = merged.get(sid)
            if prev is None:
                merged[sid] = cand
            else:
                if float(cand.get("semantic_score", 0.0)) > float(prev.get("semantic_score", 0.0)):
                    prev["semantic_score"] = float(cand.get("semantic_score", 0.0))
                if not prev.get("chunks") and cand.get("chunks"):
                    prev["chunks"] = cand.get("chunks", [])

        for cand in SPARSE_INDEX.query_schools(question=q, top_k=candidate_k):
            sid = str(cand.get("school_id", "")).strip() or str(cand.get("school", {}).get("school_id", "")).strip()
            if not sid:
                continue
            prev = merged.get(sid)
            if prev is None:
                merged[sid] = {
                    "school_id": sid,
                    "school": cand.get("school", {}),
                    "chunks": cand.get("chunks", []),
                    "text": cand.get("text", ""),
                    "semantic_score": 0.0,
                    "sparse_score": float(cand.get("sparse_score", 0.0)),
                }
            else:
                prev["sparse_score"] = max(float(prev.get("sparse_score", 0.0)), float(cand.get("sparse_score", 0.0)))
                if not prev.get("chunks") and cand.get("chunks"):
                    prev["chunks"] = cand.get("chunks", [])

    candidates = list(merged.values())
    if not candidates:
        return []

    filtered: list[dict] = []
    mentioned_school_ids = _extract_school_mentions(question, schools)
    for item in candidates:
        school = item["school"]
        chunks = item.get("chunks", [])
        if not school_matches_profile(school, profile):
            continue
        if city_intent and not _city_matches_intent(str(school.get("city", "")), city_intent):
            continue
        filtered.append(item)

    if not filtered:
        # Recall-safe fallback: keep country-matching candidates if stream constraints are too strict.
        filtered = [item for item in candidates if school_matches_profile(item["school"], profile)]
        if city_intent:
            city_filtered = [
                item
                for item in filtered
                if _city_matches_intent(str(item.get("school", {}).get("city", "")), city_intent)
            ]
            if city_filtered:
                filtered = city_filtered
    if not filtered:
        return []

    constrained = [
        item
        for item in filtered
        if _school_matches_query_constraints(item.get("school", {}), item.get("chunks", []), query_constraints)
    ]
    if constrained:
        filtered = constrained

    rescored: list[dict] = []
    strict_intent = _has_explicit_program_intent(question)
    for item in filtered:
        school = item["school"]
        chunks = item.get("chunks", [])
        semantic = float(item.get("semantic_score", 0.0))
        sparse = float(item.get("sparse_score", 0.0))
        lexical = _lexical_match_score(question, school, chunks)
        sparse_weight = min(0.7, max(0.1, _env_float("HYBRID_SPARSE_WEIGHT", 0.35)))
        dense_weight = 1.0 - sparse_weight
        hybrid_semantic = dense_weight * semantic + sparse_weight * sparse
        hybrid_semantic = 0.85 * hybrid_semantic + 0.15 * lexical
        components = _score_candidate(
            question,
            profile,
            school,
            chunks,
            hybrid_semantic,
            city_intent=city_intent,
        )
        components["lexical_match"] = lexical
        components["sparse_score"] = sparse
        components["hybrid_semantic"] = hybrid_semantic
        school_id = str(school.get("school_id", ""))
        is_mentioned = school_id in mentioned_school_ids

        # If user clearly asks for a domain, keep only reasonably aligned programs.
        if strict_intent and components["program_match"] < 0.03 and components["intent_match"] < 0.05 and hybrid_semantic < 0.08 and not is_mentioned:
            continue

        if is_mentioned:
            components["final"] += 0.12

        if location_only_query:
            breadth = _school_breadth_score(school)
            # For broad city-only intent, prefer schools with broader program coverage.
            components["final"] = 0.7 * components["final"] + 0.3 * breadth
            components["breadth_score"] = breadth

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
                "score": float(0.85 * components["final"] + 0.15 * hybrid_semantic),
                "chunk": chosen_chunk,
                "school": school,
                "score_components": components,
            }
        )

    rescored.sort(key=lambda x: x["score"], reverse=True)
    if not rescored:
        return []
    # Return up to requested top_k (capped) for stronger recall during evaluation.
    select_n = min(max(3, top_k), 20)
    return rescored[:select_n]
