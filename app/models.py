from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class UserProfile:
    bac_stream: str
    expected_grade_band: str
    motivation: str
    budget_band: str
    city: str
    country: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "UserProfile":
        return UserProfile(
            bac_stream=str(data.get("bac_stream", "")).strip().lower(),
            expected_grade_band=str(data.get("expected_grade_band", "")).strip().lower(),
            motivation=str(data.get("motivation", "")).strip().lower(),
            budget_band=str(data.get("budget_band", "")).strip().lower(),
            city=str(data.get("city", "")).strip(),
            country=str(data.get("country", "")).strip().upper(),
        )


@dataclass
class QueryRequest:
    question: str
    profile: UserProfile
    top_k: int = 5

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "QueryRequest":
        top_k = int(data.get("top_k", 5))
        top_k = max(1, min(10, top_k))
        return QueryRequest(
            question=str(data.get("question", "")).strip(),
            profile=UserProfile.from_dict(data.get("profile", {})),
            top_k=top_k,
        )


@dataclass
class EvidenceItem:
    chunk_id: str
    school_id: str
    school_name: str
    program: str
    recorded_at: str
    text: str
    score: float


@dataclass
class QueryResponse:
    short_answer: str
    why_it_fits: str
    evidence: list[EvidenceItem]
    alternative: str
    next_action: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "short_answer": self.short_answer,
            "why_it_fits": self.why_it_fits,
            "evidence": [asdict(item) for item in self.evidence],
            "alternative": self.alternative,
            "next_action": self.next_action,
            "confidence": self.confidence,
        }


@dataclass
class EvalResult:
    id: str
    passed: bool
    checks: dict[str, bool]
    answer_preview: str


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "results": [asdict(item) for item in self.results],
        }
