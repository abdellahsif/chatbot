from __future__ import annotations

from dataclasses import dataclass
import difflib
import json
import math
import os
from pathlib import Path
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
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

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


def _normalize_city_key(text: str) -> str:
    lowered = (text or "").strip().lower()
    folded = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"[^a-z0-9\s/,-]+", " ", folded)
    return " ".join(folded.split())


_DEFAULT_CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "agadir": (30.4278, -9.5981),
    "al hoceima": (35.2470, -3.9320),
    "azrou": (33.4342, -5.2213),
    "beni mellal": (32.3373, -6.3498),
    "berrechid": (33.2655, -7.5875),
    "casablanca": (33.5731, -7.5898),
    "dakhla": (23.6848, -15.9570),
    "el jadida": (33.2316, -8.5007),
    "errachidia": (31.9314, -4.4244),
    "essaouira": (31.5085, -9.7595),
    "fes": (34.0181, -5.0078),
    "fkih ben salah": (32.5000, -6.6900),
    "guelmim": (28.9870, -10.0574),
    "ifrane": (33.5333, -5.1100),
    "kenitra": (34.2610, -6.5802),
    "khemisset": (33.8246, -6.0663),
    "khenifra": (32.9349, -5.6617),
    "khouribga": (32.8830, -6.9063),
    "laayoune": (27.1536, -13.2033),
    "larache": (35.1932, -6.1563),
    "marrakech": (31.6295, -7.9811),
    "martil": (35.6167, -5.2833),
    "meknes": (33.8935, -5.5473),
    "nador": (35.1681, -2.9335),
    "ouarzazate": (30.9335, -6.9370),
    "oujda": (34.6814, -1.9086),
    "rabat": (34.0209, -6.8416),
    "sale": (34.0372, -6.7985),
    "safi": (32.2994, -9.2372),
    "sefrou": (33.8315, -4.8353),
    "settat": (33.0010, -7.6166),
    "sidi bennour": (32.6493, -8.4250),
    "tanger": (35.7595, -5.8340),
    "temara": (33.9287, -6.9063),
    "tetouan": (35.5889, -5.3626),
}

_DEFAULT_CITY_ALIASES = {
    "beni mellal khenifra": "beni mellal",
    "beni mellal-khenifra": "beni mellal",
    "beni mellal khenifra": "beni mellal",
    "casa": "casablanca",
    "casa blanca": "casablanca",
    "el jadida": "el jadida",
    "fes": "fes",
    "fes meknes": "fes",
    "fkih bensalah": "fkih ben salah",
    "fkih ben saleh": "fkih ben salah",
    "fkih ben salah": "fkih ben salah",
    "laayoune": "laayoune",
    "laayoun": "laayoune",
    "oujda angad": "oujda",
    "rabat sale kenitra": "rabat",
    "rabat-sale-kenitra": "rabat",
    "tangier": "tanger",
    "tetuan": "tetouan",
}


def _parse_city_catalog(payload: Any) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    coordinates: dict[str, tuple[float, float]] = {}
    aliases: dict[str, str] = {}

    entries: list[dict[str, Any]] = []
    if isinstance(payload, list):
        entries = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("cities"), list):
            entries = [item for item in payload["cities"] if isinstance(item, dict)]
        coord_map = payload.get("coordinates")
        if isinstance(coord_map, dict):
            for raw_name, raw_value in coord_map.items():
                if not isinstance(raw_name, str):
                    continue
                canonical = _normalize_city_key(raw_name)
                if not canonical:
                    continue
                if (
                    isinstance(raw_value, (list, tuple))
                    and len(raw_value) >= 2
                ):
                    try:
                        coordinates[canonical] = (float(raw_value[0]), float(raw_value[1]))
                    except (TypeError, ValueError):
                        continue
        alias_map = payload.get("aliases")
        if isinstance(alias_map, dict):
            for raw_alias, raw_target in alias_map.items():
                if not isinstance(raw_alias, str) or not isinstance(raw_target, str):
                    continue
                alias_key = _normalize_city_key(raw_alias)
                target_key = _normalize_city_key(raw_target)
                if alias_key and target_key:
                    aliases[alias_key] = target_key

    for item in entries:
        raw_name = (
            item.get("name")
            or item.get("city")
            or item.get("nom")
            or item.get("nom_ville")
            or ""
        )
        canonical = _normalize_city_key(str(raw_name))
        if not canonical:
            continue

        raw_lat = item.get("lat", item.get("latitude"))
        raw_lon = item.get("lon", item.get("lng", item.get("longitude")))
        if raw_lat is not None and raw_lon is not None:
            try:
                coordinates[canonical] = (float(raw_lat), float(raw_lon))
            except (TypeError, ValueError):
                pass

        raw_aliases = item.get("aliases", [])
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        if isinstance(raw_aliases, list):
            for raw_alias in raw_aliases:
                if not isinstance(raw_alias, str):
                    continue
                alias_key = _normalize_city_key(raw_alias)
                if alias_key:
                    aliases[alias_key] = canonical

    for canonical in coordinates:
        aliases.setdefault(canonical, canonical)

    return coordinates, aliases


def _load_city_catalog() -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    coordinates = dict(_DEFAULT_CITY_COORDINATES)
    aliases = dict(_DEFAULT_CITY_ALIASES)

    configured = os.getenv("MOROCCO_CITIES_JSON", "").strip()
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        Path(configured) if configured else None,
        project_root / "ma.json",
        project_root / "data" / "moroccan_cities.json",
    ]

    source_path = None
    for candidate in candidate_paths:
        if candidate is not None and candidate.exists():
            source_path = candidate
            break

    if source_path is not None:
        try:
            with source_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            parsed_coords, parsed_aliases = _parse_city_catalog(payload)
            if parsed_coords:
                coordinates = parsed_coords
            if parsed_aliases:
                for alias_key, alias_target in parsed_aliases.items():
                    aliases.setdefault(alias_key, alias_target)
        except Exception:
            pass

    for canonical in coordinates:
        aliases.setdefault(canonical, canonical)

    return coordinates, aliases


_CITY_COORDINATES, _CITY_ALIASES = _load_city_catalog()


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

        retrieve_k = max(top_k, 5)
        retrieve_k = min(retrieve_k, max(10, _env_int("DENSE_RETRIEVE_MAX", 60)))
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


class _CrossEncoderReranker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._model: Any = None
        self._model_name: str = ""

    def _get_model(self) -> Any:
        if not _env_bool("USE_CROSS_ENCODER_RERANKER", True):
            return None
        if CrossEncoder is None:
            return None

        model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        if self._model is not None and self._model_name == model_name:
            return self._model

        with self._lock:
            if self._model is not None and self._model_name == model_name:
                return self._model
            try:
                self._model = CrossEncoder(model_name)
                self._model_name = model_name
            except Exception:
                # Fail open: keep base ranking if cross-encoder cannot be loaded.
                self._model = None
                self._model_name = ""
        return self._model

    def rerank(self, question: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return candidates

        model = self._get_model()
        if model is None:
            return candidates

        top_n = max(2, _env_int("CROSS_ENCODER_TOP_N", 8))
        top_n = min(top_n, len(candidates))
        blend = min(1.0, max(0.0, _env_float("CROSS_ENCODER_BLEND", 0.6)))

        pairs: list[tuple[str, str]] = []
        for item in candidates[:top_n]:
            school = item.get("school", {})
            chunk = item.get("chunk", {})
            text = " ".join(
                [
                    str(school.get("name", "")),
                    str(school.get("city", "")),
                    str(school.get("type", "")),
                    " ".join(school.get("programs", [])),
                    str(chunk.get("program", "")),
                    str(chunk.get("text", "")),
                ]
            )
            pairs.append((question, text))

        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
        except Exception:
            return candidates

        if hasattr(raw_scores, "tolist"):
            raw_values = [float(v) for v in raw_scores.tolist()]
        else:
            raw_values = [float(v) for v in raw_scores]

        lo = min(raw_values) if raw_values else 0.0
        hi = max(raw_values) if raw_values else 0.0
        if hi > lo:
            ce_norm = [(v - lo) / (hi - lo) for v in raw_values]
        else:
            ce_norm = [1.0 / (1.0 + math.exp(-v)) for v in raw_values]

        reranked = list(candidates)
        for i in range(top_n):
            item = dict(reranked[i])
            base_score = float(item.get("score", 0.0))
            ce_score = float(ce_norm[i])
            final_score = (1.0 - blend) * base_score + blend * ce_score

            components = dict(item.get("score_components", {}))
            components["cross_encoder_score"] = ce_score
            components["cross_encoder_blend"] = blend
            item["score_components"] = components
            item["score"] = float(final_score)
            reranked[i] = item

        reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return reranked


CROSS_ENCODER_RERANKER = _CrossEncoderReranker()
MIN_SCORE_THRESHOLD = max(0.0, min(1.0, _env_float("MIN_SCORE_THRESHOLD", 0.60)))


def budget_allows(profile_budget: str, tuition_max_mad: int) -> bool:
    if not isinstance(tuition_max_mad, int):
        return False
    if profile_budget == "zero_public":
        return tuition_max_mad <= 12000
    cap = BUDGET_MAX.get(profile_budget)
    if cap is None:
        return tuition_max_mad <= BUDGET_MAX["comfort_50k"]
    return tuition_max_mad <= cap


def school_is_public(school: dict[str, Any]) -> bool:
    school_type = str(school.get("type", "")).strip().lower()
    legal_status = str(school.get("legal_status", "")).strip().lower()
    if school_type == "public":
        return True
    if any(token in legal_status for token in ["public", "publique", "etat", "state"]):
        return True
    return False


def profile_requires_public_only(profile: UserProfile) -> bool:
    return profile.budget_band == "zero_public"


def school_matches_profile(school: dict, profile: UserProfile) -> bool:
    if str(school.get("country", "")).upper() != profile.country:
        return False
    return True


def _normalize_bac_series(text: str) -> str:
    value = _normalize_city_text(text)
    if not value:
        return ""

    mapping = {
        "science mathematiques": "sm",
        "sciences mathematiques": "sm",
        "science math": "sm",
        "sciences math": "sm",
        "sm": "sm",
        "pc": "spc",
        "spc": "spc",
        "sciences physiques": "spc",
        "physique": "spc",
        "svt": "svt",
        "sciences de la vie": "svt",
        "sciences de la vie et de la terre": "svt",
        "science de la vie": "svt",
        "sciences de la vie et terre": "svt",
        "eco": "eco",
        "economique": "eco",
        "economie": "eco",
        "sciences economiques": "eco",
        "sciences de gestion": "eco",
        "lettres": "lettres",
        "litterature": "lettres",
        "sciences humaines": "lettres",
        "sciences humaines et sociales": "lettres",
        "sciences sociales": "lettres",
        "humanities": "lettres",
        "arts": "arts",
        "art": "arts",
        "design": "arts",
    }
    return mapping.get(value, value)


_BAC_SERIES_ALLOWED_DOMAINS: dict[str, set[str]] = {
    "sm": {"engineering", "computer", "science", "health", "business"},
    "spc": {"engineering", "computer", "science", "business"},
    "svt": {"health", "science"},
    "eco": {"business", "law", "public_admin"},
    "lettres": {"law", "arts", "humanities", "business"},
    "arts": {"arts", "humanities"},
}

_SCHOOL_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "military": {"military", "army", "defense", "defence", "gendarmerie", "royale", "parachut", "infanterie", "blind", "transport"},
    "business": {"business", "commerce", "management", "gestion", "finance", "marketing", "entreprise", "accounting", "audit", "bank"},
    "engineering": {"engineering", "ingenieur", "ingenierie", "tech", "technologie", "science", "appliquee", "appliquees", "telecommunication", "telecommunications", "tic", "numerique", "digital"},
    "computer": {"informatique", "software", "computer", "cs", "data", "cyber", "programming", "code", "developpement", "telecom", "telecommunication", "telecommunications", "tic", "numerique", "digital"},
    "health": {"health", "sante", "medical", "medecine", "pharmacie", "paramedical", "nursing", "biologie"},
    "arts": {"art", "arts", "design", "beaux", "creative", "cinema", "architecture", "portfolio"},
    "law": {"law", "droit", "legal", "juridique"},
    "humanities": {"lettres", "litterature", "humanities", "communication", "education", "sharia", "philosophie"},
    "public_admin": {"administration", "public", "gouvernance", "policy", "police", "diplomatie"},
    "science": {"science", "sciences", "research", "recherche"},
}


def _categories_from_tokens(tokens: set[str]) -> set[str]:
    categories: set[str] = set()
    if not tokens:
        return categories

    for category, keywords in _SCHOOL_DOMAIN_KEYWORDS.items():
        if tokens & keywords:
            categories.add(category)

    technical_domains = {"engineering", "computer", "science", "health", "military"}
    if categories & technical_domains:
        categories.discard("humanities")
        categories.discard("public_admin")
    return categories


def _school_domain_categories(school: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    # Strict rule: determine academic domain from schools.filieres first.
    text = " ".join(
        [
            str(school.get("filieres", "")),
            str(school.get("programs_tags", "")),
        ]
    )
    tokens = _tokenize(text)
    if not tokens:
        # Fallback for sparse sources: infer categories from programs and snippets.
        fallback_text = " ".join(
            [
                str(school.get("name", "")),
                str(school.get("type", "")),
                " ".join(str(p) for p in school.get("programs", [])),
                " ".join(str(c.get("program", "")) for c in chunks[:6]),
                " ".join(str(c.get("text", "")) for c in chunks[:2]),
            ]
        )
        tokens = _tokenize(fallback_text)
    if not tokens:
        return set()

    return _categories_from_tokens(tokens)


def _school_bac_compatible(profile_bac_stream: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> bool:
    bac = _normalize_bac_series(profile_bac_stream)
    if not bac:
        return True

    allowed_domains = _BAC_SERIES_ALLOWED_DOMAINS.get(bac)
    if not allowed_domains:
        return True

    school_domains = _school_domain_categories(school, chunks)
    if not school_domains:
        return False

    # Require a real semantic overlap with the school's academic domain.
    return bool(school_domains & allowed_domains)


def _has_semantic_domain_incompatibility(profile_bac_stream: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> bool:
    """Hard reject schools whose filiere domain is fundamentally incompatible with bac stream."""
    bac = _normalize_bac_series(profile_bac_stream)
    if not bac:
        return False

    school_domains = _school_domain_categories(school, chunks)
    if not school_domains:
        return True

    # Explicit strict rejects from product rules.
    if bac == "eco" and "military" in school_domains:
        return True
    if bac == "svt" and ({"engineering", "computer", "military"} & school_domains):
        return True
    if bac == "spc" and ({"law", "military"} & school_domains):
        return True
    if bac == "lettres" and ({"engineering", "computer", "health", "military", "science"} & school_domains):
        return True

    allowed_domains = _BAC_SERIES_ALLOWED_DOMAINS.get(bac)
    if not allowed_domains:
        return False
    return not bool(school_domains & allowed_domains)


def _bac_semantic_score(profile_bac_stream: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    bac = _normalize_bac_series(profile_bac_stream)
    if not bac:
        return 1.0

    school_tokens = _tokenize(
        " ".join(
            [
                str(school.get("filieres", "")),
                str(school.get("programs_tags", "")),
            ]
        )
    )
    if not school_tokens:
        return 0.0

    keyword_groups: dict[str, list[set[str]]] = {
        "eco": [
            {"business", "commerce", "management", "gestion", "finance", "accounting", "audit"},
            {"law", "droit", "juridique", "legal"},
            {"economics", "economie", "economique", "public", "administration"},
        ],
        "sm": [
            {"engineering", "ingenieur", "ingenierie", "tech", "technologie"},
            {"computer", "informatique", "software", "data", "cyber", "programming", "code"},
            {"science", "sciences", "math", "mathematiques"},
        ],
        "spc": [
            {"engineering", "ingenieur", "ingenierie", "tech", "technologie"},
            {"computer", "informatique", "software", "data", "cyber", "programming", "code"},
            {"science", "sciences", "physique", "physiques"},
        ],
        "svt": [
            {"health", "sante", "medical", "medecine", "pharmacie", "paramedical", "nursing"},
            {"biology", "biologie", "bio", "life", "vie", "terre"},
            {"science", "sciences", "environment", "ecologie"},
        ],
        "lettres": [
            {"arts", "art", "design", "creative", "cinema"},
            {"humanities", "lettres", "litterature", "communication", "education"},
        ],
        "arts": [
            {"arts", "art", "design", "creative", "cinema", "portfolio"},
            {"architecture", "urbanisme", "beaux", "graphique"},
        ],
    }

    groups = keyword_groups.get(bac)
    if not groups:
        return 0.7 if _school_bac_compatible(profile_bac_stream, school, chunks) else 0.0

    hits: list[float] = []
    for group in groups:
        overlap = len(school_tokens & group)
        if overlap:
            hits.append(min(1.0, overlap / float(len(group))))

    if not hits:
        return 0.0

    return min(1.0, max(hits) * 0.65 + sum(hits) / len(hits) * 0.35)


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
    "international": {"international", "global", "exposure", "abroad", "expat", "double", "degree"},
}

DOMAIN_TERMS: dict[str, set[str]] = {
    "law": {"law", "droit", "juridique", "legal", "jurisprudence"},
    "medicine": {"medicine", "medical", "medecine", "pharmacie"},
    "healthcare": {"health", "healthcare", "sante", "paramedical", "nursing", "care", "sanitaire"},
    "computer": {"computer", "informatique", "software", "developpement", "programmation", "data", "cyber", "ai", "ia"},
    "engineering": {"engineering", "ingenieur", "ingenierie", "technologie", "tech"},
    "business": {"business", "commerce", "gestion", "finance", "management", "economie"},
    "arts": {"art", "arts", "design", "beaux", "cinema"},
    "military": {"military", "armee", "defense", "gendarmerie", "royale", "infanterie", "blindes", "artillerie", "marines", "parachutistes"},
}

EXPLICIT_DOMAIN_TO_CATEGORIES: dict[str, set[str]] = {
    "law": {"law"},
    "medicine": {"health"},
    "healthcare": {"health"},
    "computer": {"computer"},
    "engineering": {"engineering", "computer"},
    "business": {"business", "public_admin"},
    "arts": {"arts", "humanities"},
    "military": {"military"},
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
            str(school.get("filieres", "")),
            str(school.get("programs_tags", "")),
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


def _school_matches_explicit_domains(
    school: dict[str, Any],
    chunks: list[dict[str, Any]],
    domains: set[str],
) -> bool:
    if not domains:
        return True

    required_categories: set[str] = set()
    for domain in domains:
        required_categories |= EXPLICIT_DOMAIN_TO_CATEGORIES.get(domain, set())

    primary_tokens = _tokenize(
        " ".join(
            [
                str(school.get("filieres", "")),
                str(school.get("programs_tags", "")),
            ]
        )
    )
    primary_categories = _categories_from_tokens(primary_tokens)

    fallback_tokens = _tokenize(
        " ".join(
            [
                str(school.get("name", "")),
                str(school.get("type", "")),
                " ".join(str(p) for p in school.get("programs", [])),
            ]
        )
    )
    fallback_categories = _categories_from_tokens(fallback_tokens)

    school_categories = primary_categories or fallback_categories
    if required_categories:
        # Hard gate: for mapped explicit domains, require category overlap from filieres/programs_tags.
        if not school_categories:
            return False
        return bool(school_categories & required_categories)

    domain_tokens = _school_domain_tokens(school, chunks)
    if not domain_tokens:
        return False
    for domain in domains:
        terms = DOMAIN_TERMS.get(domain, set())
        if terms and (domain_tokens & terms):
            return True
    return False


def _normalize_city_text(text: str) -> str:
    return _normalize_city_key(text)


def extract_cities(ville_field: str) -> list[str]:
    if not ville_field:
        return []
    cities = re.split(r"[/\-,]", str(ville_field))
    return [city.strip() for city in cities if city and city.strip()]


def _canonical_city_name(city: str) -> str:
    base = _normalize_city_text(city)
    return _CITY_ALIASES.get(base, base)


def _iter_city_tokens(ville_field: str) -> list[str]:
    normalized = _normalize_city_text(ville_field)
    if not normalized:
        return []

    out: list[str] = []
    for city in extract_cities(normalized):
        canonical = _canonical_city_name(city)
        if canonical and canonical not in out:
            out.append(canonical)
    if not out:
        canonical = _canonical_city_name(normalized)
        if canonical:
            out.append(canonical)
    return out


def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _distance_between_cities_km(city_a: str, city_b: str) -> float | None:
    a = _CITY_COORDINATES.get(_canonical_city_name(city_a))
    b = _CITY_COORDINATES.get(_canonical_city_name(city_b))
    if not a or not b:
        return None
    return _haversine_distance_km(a[0], a[1], b[0], b[1])


def _distance_to_school_city_km(target_city: str, school_city: str) -> float | None:
    dists: list[float] = []
    for candidate in _iter_city_tokens(school_city):
        d = _distance_between_cities_km(target_city, candidate)
        if d is not None:
            dists.append(d)
    if not dists:
        return None
    return min(dists)


def _available_school_cities(schools: dict[str, dict]) -> set[str]:
    out: set[str] = set()
    for school in schools.values():
        out.update(_iter_city_tokens(str(school.get("city", ""))))
    return out


def _nearest_cities_from_target(target_city: str, schools: dict[str, dict], limit: int = 5) -> list[tuple[str, float]]:
    canonical_target = _canonical_city_name(target_city)
    if not canonical_target:
        return []

    nearest: list[tuple[str, float]] = []
    for city in _available_school_cities(schools):
        if city == canonical_target:
            continue
        dist = _distance_between_cities_km(canonical_target, city)
        if dist is not None:
            nearest.append((city, dist))

    nearest.sort(key=lambda x: x[1])
    return nearest[: max(1, limit)]


def _school_matches_any_city(school_city: str, candidate_cities: set[str]) -> bool:
    if not candidate_cities:
        return False

    school_tokens = set(_iter_city_tokens(school_city))
    if school_tokens & candidate_cities:
        return True

    for school_token in school_tokens:
        for candidate in candidate_cities:
            if difflib.SequenceMatcher(a=school_token, b=candidate).ratio() >= 0.88:
                return True
    return False


def _extract_city_intent(question: str, schools: dict[str, dict]) -> str | None:
    q = _normalize_city_text(question)
    if not q:
        return None

    known_cities = _available_school_cities(schools)

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


def _extract_budget_override(question: str) -> str | None:
    q = _normalize_city_text(question)
    if not q:
        return None

    if re.search(r"\b(no\s*limit|unlimited|sans\s*limite|illimite)\b", q):
        return "no_limit_70k_plus"
    if re.search(r"\b(70k|70000|70\s*000)\b", q):
        return "no_limit_70k_plus"
    if re.search(r"\b(50k|50000|50\s*000)\b", q):
        return "comfort_50k"
    if re.search(r"\b(25k|25000|25\s*000)\b", q):
        return "tight_25k"
    if re.search(r"\b(public\s*only|zero\s*budget|free|gratuit|very\s*low\s*budget|budget\s*zero)\b", q):
        return "zero_public"
    if re.search(r"\b(affordable|cheap|low\s*cost|pas\s*cher|economique)\b", q):
        return "tight_25k"
    return None


def _extract_motivation_override(question: str) -> str | None:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return None

    rules: list[tuple[str, set[str]]] = [
        ("expat", {"abroad", "expat", "international", "global", "leave", "outside"}),
        ("prestige", {"prestige", "elite", "top", "ranking", "reputation"}),
        ("cash", {"budget", "cheap", "affordable", "cost", "roi", "salary", "income"}),
        ("employability", {"job", "jobs", "work", "employability", "hire", "career"}),
        ("safety", {"safe", "safer", "stable", "realistic", "fallback"}),
        ("passion", {"passion", "love", "interested", "interest"}),
    ]
    best = None
    best_hits = 0
    for label, vocab in rules:
        hits = len(q_tokens & vocab)
        if hits > best_hits:
            best_hits = hits
            best = label
    return best if best_hits > 0 else None


def _extract_country_override(question: str) -> str | None:
    q = _normalize_city_text(question)
    if re.search(r"\b(morocco|maroc|ma)\b", q):
        return "MA"
    if re.search(r"\b(senegal|sn)\b", q):
        return "SN"
    if re.search(r"\b(cote\s*d\s*ivoire|ivory\s*coast|ci)\b", q):
        return "CI"
    return None


def _extract_bac_stream_override(question: str) -> str | None:
    q = _normalize_city_text(question)
    if re.search(r"\b(science\s*math|sm)\b", q):
        return "sm"
    if re.search(r"\b(physics|pc|spc|physique)\b", q):
        return "spc"
    if re.search(r"\b(svt|bio|biology)\b", q):
        return "svt"
    if re.search(r"\b(eco|economics|economie)\b", q):
        return "eco"
    if re.search(r"\b(lettres|literature|humanities)\b", q):
        return "lettres"
    return None


def _extract_grade_override(question: str) -> str | None:
    q = _normalize_city_text(question)
    if re.search(r"\b(elite|excellent|16\s*20|16\s*/\s*20|17\s*/\s*20|18\s*/\s*20|19\s*/\s*20|20\s*/\s*20)\b", q):
        return "elite"
    if re.search(r"\b(tres\s*bien|14\s*16|14\s*/\s*20|15\s*/\s*20|16\s*/\s*20)\b", q):
        return "tres_bien"
    if re.search(r"\b(bien|12\s*14|12\s*/\s*20|13\s*/\s*20|14\s*/\s*20)\b", q):
        return "bien"
    if re.search(r"\b(passable|10\s*12|10\s*/\s*20|11\s*/\s*20|12\s*/\s*20)\b", q):
        return "passable"
    return None


def resolve_effective_profile(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
) -> UserProfile:
    city_intent = _extract_city_intent(question, schools)
    budget_override = _extract_budget_override(question)
    motivation_override = _extract_motivation_override(question)
    country_override = _extract_country_override(question)
    bac_override = _extract_bac_stream_override(question)
    grade_override = _extract_grade_override(question)

    if not any([city_intent, budget_override, motivation_override, country_override, bac_override, grade_override]):
        return profile

    return UserProfile(
        bac_stream=bac_override or profile.bac_stream,
        expected_grade_band=grade_override or profile.expected_grade_band,
        motivation=motivation_override or profile.motivation,
        budget_band=budget_override or profile.budget_band,
        city=city_intent or profile.city,
        country=(country_override or profile.country).upper(),
    )


def _city_matches_intent(school_city: str, city_intent: str | None) -> bool:
    if not city_intent:
        return True
    target = _canonical_city_name(city_intent)
    candidates = set(_iter_city_tokens(school_city))
    if not candidates:
        return False

    for candidate in candidates:
        if candidate == target:
            return True
        if target in candidate or candidate in target:
            return True
        if difflib.SequenceMatcher(a=candidate, b=target).ratio() >= 0.84:
            return True
    return False


def _has_explicit_city_constraint(question: str, city_intent: str | None) -> bool:
    if not city_intent:
        return False
    q = _normalize_city_text(question)
    if not q:
        return False
    if re.search(r"\b(near|around|proche)\b", q):
        return False
    city_pat = re.escape(city_intent)
    return bool(re.search(rf"\b(in|at|en|au|aux|dans)\s+{city_pat}\b", q))


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


def _sanitize_query_text(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return ""

    patterns = [
        r"^please answer in english only\.?\s*",
        r"^reponds en francais:?\s*",
        r"^jawbni b darija:?\s*",
        r"^give me a strict recommendation with tradeoffs\.?\s*",
        r"^edge case:\s*very low budget and uncertain grades\.?\s*",
        r"^keep answer grounded in evidence only\.?\s*",
    ]

    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new_text = re.sub(pat, "", text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text.strip()
                changed = True

    typo_map = {
        "affrodable": "affordable",
        "whch": "which",
        "optns": "options",
        "internatonal": "international",
        "managment": "management",
        "enginering": "engineering",
        "universty": "university",
        "scholl": "school",
        "agadeer": "agadir",
    }

    words = text.split()
    normalized_words: list[str] = []
    for word in words:
        leading = ""
        trailing = ""
        core = word
        while core and not core[0].isalnum():
            leading += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            trailing = core[-1] + trailing
            core = core[:-1]
        replacement = typo_map.get(core.lower(), core)
        normalized_words.append(f"{leading}{replacement}{trailing}")

    text = " ".join(normalized_words)
    text = re.sub(r"\s+", " ", text).strip()
    return text or (question or "").strip()


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
    return _school_bac_compatible(profile_bac_stream, school, chunks)


def _bac_stream_match_score(profile_bac_stream: str, chunks: list[dict[str, Any]], school: dict[str, Any]) -> float:
    return 1.0 if _bac_stream_compatible(profile_bac_stream, chunks, school) else 0.0


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
            str(school.get("international_double_degree", "")),
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
    raw_q_tokens = _tokenize(question)
    q_tokens = _expanded_query_tokens(question)
    intent_tokens = set().union(*INTENT_SYNONYMS.values())
    matched_groups = [group for group in INTENT_SYNONYMS.values() if raw_q_tokens & group]
    strict_intent = _has_explicit_program_intent(question)
    if strict_intent and matched_groups:
        q_tokens = set().union(*matched_groups)
    if not q_tokens:
        return 0.0
    school_text = " ".join(school.get("programs", []))
    chunk_text = " ".join(
        f"{str(c.get('program', ''))} {str(c.get('text', ''))}"
        for c in chunks[:3]
    )
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


def _location_match_score(
    profile: UserProfile,
    school: dict[str, Any],
    city_intent: str | None = None,
    fallback_cities: set[str] | None = None,
) -> float:
    school_city = str(school.get("city", "")).strip()
    if city_intent:
        return 1.0 if _city_matches_intent(school_city, city_intent) else 0.0

    if fallback_cities and _school_matches_any_city(school_city, fallback_cities):
        return 0.85

    profile_city = _canonical_city_name(profile.city)
    if profile_city:
        distance_km = _distance_to_school_city_km(profile_city, school_city)
        if distance_km is not None:
            if distance_km <= 20:
                return 0.9
            if distance_km <= 50:
                return 0.8
            if distance_km <= 100:
                return 0.7
            if distance_km <= 180:
                return 0.58
            if distance_km <= 300:
                return 0.45
            if distance_km <= 500:
                return 0.35

    if profile.city and _city_matches_intent(school_city, _canonical_city_name(profile.city)):
        return 0.35
    if profile.city and school_city.lower() == profile.city.strip().lower():
        return 0.35
    if str(school.get("country", "")).upper() == profile.country:
        return 0.25
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
        intl_bonus = 0.2 if has_international else 0.0
        return min(1.0, 0.5 * selectivity_bonus + 0.3 * min(1.0, employability / 5.0) + intl_bonus)
    if motivation == "expat":
        return 1.0 if has_international else 0.25
    if motivation == "employability":
        return min(1.0, employability / 5.0)
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
    fallback_cities: set[str] | None = None,
) -> dict[str, float]:
    program_match = _program_match_score(question, school, chunks)
    intent_match = _intent_group_match_score(question, school, chunks)
    bac_match = _bac_stream_match_score(profile.bac_stream, chunks, school)
    bac_semantic = _bac_semantic_score(profile.bac_stream, school, chunks)
    budget_match = _budget_match_score(profile, school)
    grade_match = _grade_match_score(profile, school)
    location_match = _location_match_score(profile, school, city_intent=city_intent, fallback_cities=fallback_cities)
    motivation_match = _motivation_match_score(profile, school)

    weighted = (
        0.5 * bac_semantic
        + 0.2 * location_match
        + 0.15 * budget_match
        + 0.15 * motivation_match
    )
    final_score = 0.7 * weighted + 0.3 * max(0.0, semantic)
    return {
        "program_match": program_match,
        "intent_match": intent_match,
        "bac_match": bac_match,
        "bac_semantic": bac_semantic,
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


def _name_query_match_score(question: str, school_name: str) -> float:
    q_tokens = _tokenize(question)
    n_tokens = _tokenize(school_name)
    if not q_tokens or not n_tokens:
        return 0.0

    overlap = _safe_div(len(q_tokens & n_tokens), max(1, len(n_tokens)))
    acronym = _acronym_from_name(school_name)
    q_norm = _normalize_city_text(question)
    acronym_hit = 1.0 if acronym and acronym in q_norm else 0.0
    return min(1.0, max(overlap, acronym_hit))


def _tech_path_bonus(question: str, school: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0

    software_like = INTENT_SYNONYMS.get("software", set()) | INTENT_SYNONYMS.get("data", set()) | INTENT_SYNONYMS.get("cyber", set())
    if not (q_tokens & software_like):
        return 0.0

    school_text = " ".join(
        [
            str(school.get("name", "")),
            " ".join(school.get("programs", [])),
            " ".join(str(c.get("program", "")) for c in chunks[:6]),
            " ".join(str(c.get("text", "")) for c in chunks[:2]),
        ]
    )
    st = _tokenize(school_text)
    core_terms = {"informatique", "software", "computer", "programmation", "code", "developpement", "cyber"}
    support_terms = {"ingenieur", "ingenierie", "technologie", "techniques", "appliquees", "science", "sciences"}

    core_hits = len(st & core_terms)
    support_hits = len(st & support_terms)
    if core_hits >= 2:
        return 0.08
    if core_hits >= 1 and support_hits >= 1:
        return 0.06
    if core_hits >= 1 or support_hits >= 2:
        return 0.04
    return 0.0


def _affordable_public_tech_bonus(
    question: str,
    profile: UserProfile,
    school: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0

    software_like = INTENT_SYNONYMS.get("software", set()) | INTENT_SYNONYMS.get("data", set()) | INTENT_SYNONYMS.get("cyber", set())
    if not (q_tokens & software_like):
        return 0.0

    if profile.budget_band not in {"zero_public", "tight_25k"}:
        return 0.0
    if str(school.get("type", "")).strip().lower() != "public":
        return 0.0

    school_text = " ".join(
        [
            str(school.get("name", "")),
            " ".join(school.get("programs", [])),
            " ".join(str(c.get("program", "")) for c in chunks[:5]),
        ]
    )
    st = _tokenize(school_text)
    public_tech_terms = {"faculte", "sciences", "techniques", "est", "ofppt", "informatique", "appliquees"}
    hits = len(st & public_tech_terms)
    if hits >= 3:
        return 0.05
    if hits >= 2:
        return 0.035
    return 0.0


def retrieve(
    *,
    question: str,
    profile: UserProfile,
    schools: dict[str, dict],
    transcripts: list[dict],
    top_k: int,
) -> list[dict]:
    query_text = _sanitize_query_text(question)
    profile = resolve_effective_profile(
        question=query_text,
        profile=profile,
        schools=schools,
    )
    SEMANTIC_INDEX.ensure(schools, transcripts)
    SPARSE_INDEX.ensure(schools, transcripts)
    city_intent = _extract_city_intent(query_text, schools)
    profile_city = _canonical_city_name(profile.city)
    fallback_city_distances: list[tuple[str, float]] = []
    fallback_cities: set[str] = set()
    if not city_intent and profile_city:
        fallback_city_distances = _nearest_cities_from_target(profile_city, schools, limit=5)
        fallback_cities = {city for city, _ in fallback_city_distances[:3]}

    q_norm = _normalize_city_text(query_text)
    near_city_query = bool(re.search(r"\b(near|around|proche)\b", q_norm))
    query_constraints = _extract_query_constraints(query_text)
    location_only_query = _is_location_only_query(query_text, city_intent)
    explicit_city_constraint = _has_explicit_city_constraint(query_text, city_intent)
    candidate_k = max(40, min(120, top_k * 12))
    variants = _build_query_variants(query_text, profile)

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
    mentioned_school_ids = _extract_school_mentions(query_text, schools)
    for item in candidates:
        school = item["school"]
        chunks = item.get("chunks", [])
        if _has_semantic_domain_incompatibility(profile.bac_stream, school, chunks):
            continue
        if not school_matches_profile(school, profile):
            continue
        if profile_requires_public_only(profile) and not school_is_public(school):
            continue
        filtered.append(item)

    if city_intent and (location_only_query or explicit_city_constraint):
        city_filtered = [
            item
            for item in filtered
            if _city_matches_intent(str(item.get("school", {}).get("city", "")), city_intent)
        ]
        if city_filtered:
            filtered = city_filtered

    if not city_intent and profile_city and fallback_cities:
        nearest_filtered = [
            item
            for item in filtered
            if _school_matches_any_city(str(item.get("school", {}).get("city", "")), fallback_cities)
        ]
        if nearest_filtered:
            target_n = min(max(1, top_k), 20)
            if len(nearest_filtered) >= target_n:
                filtered = nearest_filtered
            else:
                nearest_ids = {
                    str(item.get("school", {}).get("school_id", ""))
                    for item in nearest_filtered
                    if str(item.get("school", {}).get("school_id", ""))
                }
                remainder = [
                    item
                    for item in filtered
                    if str(item.get("school", {}).get("school_id", "")) not in nearest_ids
                ]
                filtered = nearest_filtered + remainder

    if not filtered:
        return []

    explicit_domains: set[str] = query_constraints.get("domains", set())
    strict_intent = _has_explicit_program_intent(query_text) or bool(explicit_domains)

    if explicit_domains:
        domain_filtered = [
            item
            for item in filtered
            if _school_matches_explicit_domains(item.get("school", {}), item.get("chunks", []), explicit_domains)
        ]
        if not domain_filtered:
            return []
        filtered = domain_filtered

    constraint_hits = {
        str(item.get("school", {}).get("school_id", "")): _school_matches_query_constraints(
            item.get("school", {}),
            item.get("chunks", []),
            query_constraints,
        )
        for item in filtered
    }

    rescored: list[dict] = []
    for item in filtered:
        school = item["school"]
        chunks = item.get("chunks", [])
        semantic = float(item.get("semantic_score", 0.0))
        sparse = float(item.get("sparse_score", 0.0))
        lexical = _lexical_match_score(query_text, school, chunks)
        name_query_match = _name_query_match_score(query_text, str(school.get("name", "")))
        sparse_weight = min(0.7, max(0.1, _env_float("HYBRID_SPARSE_WEIGHT", 0.35)))
        if strict_intent:
            sparse_weight = min(0.7, max(sparse_weight, 0.5))
        dense_weight = 1.0 - sparse_weight
        hybrid_semantic = dense_weight * semantic + sparse_weight * sparse
        if strict_intent:
            hybrid_semantic = 0.72 * hybrid_semantic + 0.2 * lexical + 0.08 * name_query_match
        else:
            hybrid_semantic = 0.8 * hybrid_semantic + 0.12 * lexical + 0.08 * name_query_match
        components = _score_candidate(
            query_text,
            profile,
            school,
            chunks,
            hybrid_semantic,
            city_intent=city_intent,
            fallback_cities=fallback_cities if not city_intent else None,
        )
        components["lexical_match"] = lexical
        components["name_query_match"] = name_query_match
        components["sparse_score"] = sparse
        components["hybrid_semantic"] = hybrid_semantic
        distance_km = _distance_to_school_city_km(profile_city, str(school.get("city", ""))) if profile_city else None
        if distance_km is not None:
            components["distance_km"] = round(distance_km, 1)

        school_id = str(school.get("school_id", ""))
        is_mentioned = school_id in mentioned_school_ids
        sid = str(school.get("school_id", ""))
        matches_constraints = constraint_hits.get(sid, True)
        matches_city_intent = _city_matches_intent(str(school.get("city", "")), city_intent) if city_intent else True
        if strict_intent and query_constraints.get("has_constraints", False) and not matches_constraints and not is_mentioned:
            # Hard filter for explicit domain intent: if user asks a domain, do not substitute unrelated schools.
            continue
        if query_constraints.get("has_constraints", False):
            # Favor constraint-aligned schools without hard-pruning recall.
            components["final"] += 0.08 if matches_constraints else -0.03
        components["constraint_match"] = 1.0 if matches_constraints else 0.0
        if city_intent:
            # Question city has precedence: strong boost when city matches, soft penalty otherwise.
            if near_city_query:
                components["final"] += 0.10 if matches_city_intent else 0.0
            else:
                components["final"] += 0.16 if matches_city_intent else -0.05
        elif distance_km is not None:
            # If exact city is unavailable, softly prioritize nearest available cities.
            if distance_km <= 50:
                components["final"] += 0.08
            elif distance_km <= 100:
                components["final"] += 0.05
            elif distance_km <= 180:
                components["final"] += 0.03
            elif distance_km <= 300:
                components["final"] -= 0.03
            else:
                components["final"] -= 0.08

        if fallback_city_distances:
            components["nearest_city_fallback"] = [
                {"city": city, "distance_km": round(dist, 1)}
                for city, dist in fallback_city_distances[:3]
            ]
        components["city_intent_match"] = 1.0 if matches_city_intent else 0.0
        tech_bonus = _tech_path_bonus(query_text, school, chunks)
        components["final"] += tech_bonus
        components["tech_path_bonus"] = tech_bonus
        affordable_bonus = _affordable_public_tech_bonus(query_text, profile, school, chunks)
        components["final"] += affordable_bonus
        components["affordable_public_tech_bonus"] = affordable_bonus

        # If user clearly asks for a domain, keep only reasonably aligned programs.
        if strict_intent and not is_mentioned:
            very_weak_intent = components["program_match"] < 0.02 and components["intent_match"] < 0.08
            weak_surface_alignment = lexical < 0.06 and name_query_match < 0.06
            if very_weak_intent and weak_surface_alignment:
                continue

        if is_mentioned:
            components["final"] += 0.28

        if location_only_query:
            breadth = _school_breadth_score(school)
            # For broad city-only intent, prefer schools with broader program coverage.
            components["final"] = 0.7 * components["final"] + 0.3 * breadth
            components["breadth_score"] = breadth

        evidence_chunks = sorted(
            chunks,
            key=lambda c: _program_match_score(query_text, school, [c]),
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
    rescored = CROSS_ENCODER_RERANKER.rerank(query_text, rescored)
    if not rescored:
        return []

    threshold_filtered = [
        item
        for item in rescored
        if float(item.get("score", 0.0)) >= MIN_SCORE_THRESHOLD
    ]
    select_n = min(max(1, top_k), 20)
    if not threshold_filtered:
        return rescored[:select_n]

    if len(threshold_filtered) >= select_n:
        return threshold_filtered[:select_n]

    # Backfill with the strongest remaining candidates so API can honor top_k when data is sparse.
    selected = list(threshold_filtered)
    selected_ids = {
        str(item.get("school", {}).get("school_id", ""))
        for item in selected
        if str(item.get("school", {}).get("school_id", ""))
    }
    for item in rescored:
        sid = str(item.get("school", {}).get("school_id", ""))
        if sid and sid in selected_ids:
            continue
        selected.append(item)
        if sid:
            selected_ids.add(sid)
        if len(selected) >= select_n:
            break

    return selected[:select_n]
