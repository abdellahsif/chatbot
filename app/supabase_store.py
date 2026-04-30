from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request


@dataclass
class SupabaseConfig:
    url: str
    api_key: str
    timeout_seconds: float


def _load_config_from_env() -> SupabaseConfig | None:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    api_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not url or not api_key:
        return None

    try:
        timeout_seconds = float(os.getenv("SUPABASE_TIMEOUT_SECONDS", "10.0") or "10.0")
    except ValueError:
        timeout_seconds = 10.0

    return SupabaseConfig(
        url=url,
        api_key=api_key,
        timeout_seconds=max(1.0, timeout_seconds),
    )


def _build_headers(cfg: SupabaseConfig, prefer: str | None = None) -> dict[str, str]:
    headers = {
        "apikey": cfg.api_key,
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _rest_url(cfg: SupabaseConfig, path: str, query: str = "") -> str:
    base = f"{cfg.url}/rest/v1/{path.lstrip('/')}"
    if query:
        return f"{base}?{query}"
    return base


def _request_json(
    cfg: SupabaseConfig,
    *,
    method: str,
    path: str,
    payload: dict[str, Any] | list[dict[str, Any]] | None = None,
    query: str = "",
    prefer: str | None = None,
) -> tuple[int, Any]:
    if method.upper() != "GET":
        raise RuntimeError("Supabase store is configured as read-only; only GET is allowed")
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        _rest_url(cfg, path, query=query),
        data=body,
        headers=_build_headers(cfg, prefer=prefer),
        method=method,
    )
    try:
        with request.urlopen(req, timeout=cfg.timeout_seconds) as response:
            raw = response.read()
            if not raw:
                return response.status, None
            return response.status, json.loads(raw.decode("utf-8"))
    except error.HTTPError as http_err:
        details = http_err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Supabase HTTP {http_err.code}: {details}") from http_err
    except error.URLError as url_err:
        raise RuntimeError(f"Supabase network error: {url_err}") from url_err


def fetch_recent_eval_runs(limit: int = 20) -> dict[str, Any]:
    table = (os.getenv("SUPABASE_EVAL_TABLE") or "eval_runs").strip()
    payload_column = (os.getenv("SUPABASE_EVAL_PAYLOAD_COLUMN") or "payload").strip()
    order_column = (os.getenv("SUPABASE_EVAL_ORDER_COLUMN") or "").strip()

    cfg = _load_config_from_env()
    if cfg is None:
        return {"enabled": False, "items": [], "reason": "missing_supabase_env"}

    safe_limit = max(1, min(200, int(limit)))
    params = {
        "select": payload_column,
        "limit": str(safe_limit),
    }
    if order_column:
        params["order"] = f"{order_column}.desc"

    status, rows = _request_json(
        cfg,
        method="GET",
        path=table,
        query=parse.urlencode(params),
    )

    items: list[dict[str, Any]] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                payload = row.get(payload_column)
                if isinstance(payload, dict):
                    items.append(payload)

    return {
        "enabled": True,
        "status": status,
        "items": items,
        "table": table,
        "limit": safe_limit,
    }


def fetch_table_rows(
    table: str,
    *,
    limit: int = 100,
    order_column: str = "",
    select: str = "*",
) -> dict[str, Any]:
    cfg = _load_config_from_env()
    if cfg is None:
        return {"enabled": False, "items": [], "reason": "missing_supabase_env"}

    safe_table = (table or "").strip()
    if not safe_table:
        return {"enabled": True, "items": [], "reason": "missing_table_name"}

    safe_limit = max(1, min(500, int(limit)))
    params = {
        "select": select or "*",
        "limit": str(safe_limit),
    }
    if order_column:
        params["order"] = f"{order_column}.desc"

    status, rows = _request_json(
        cfg,
        method="GET",
        path=safe_table,
        query=parse.urlencode(params),
    )

    items = rows if isinstance(rows, list) else []
    return {
        "enabled": True,
        "status": status,
        "items": items,
        "table": safe_table,
        "limit": safe_limit,
    }


def fetch_schools(limit: int = 100) -> dict[str, Any]:
    table = (os.getenv("SUPABASE_TABLE_SCHOOLS") or "schools").strip()
    order_column = (os.getenv("SUPABASE_SCHOOLS_ORDER_COLUMN") or "").strip()
    select = (os.getenv("SUPABASE_SCHOOLS_SELECT") or "*").strip()
    return fetch_table_rows(
        table,
        limit=limit,
        order_column=order_column,
        select=select,
    )


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def fetch_user_career_profile(user_id: str) -> dict[str, Any] | None:
    safe_user_id = str(user_id or "").strip()
    if not safe_user_id:
        return None

    cfg = _load_config_from_env()
    if cfg is None:
        return None

    table = (os.getenv("SUPABASE_TABLE_USER_CAREER_PROFILES") or "user_career_profiles").strip()
    params = {
        "select": "id,userId,inferredCareers,domainScores,computed_at",
        "userId": f"eq.{safe_user_id}",
        "order": "computed_at.desc",
        "limit": "1",
    }

    _, rows = _request_json(
        cfg,
        method="GET",
        path=table,
        query=parse.urlencode(params),
    )

    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0] if isinstance(rows[0], dict) else None
    if not row:
        return None

    inferred_raw = _parse_json_like(row.get("inferredCareers"))
    inferred_careers: list[str] = []
    if isinstance(inferred_raw, list):
        inferred_careers = [str(item).strip().lower() for item in inferred_raw if str(item).strip()]

    domain_raw = _parse_json_like(row.get("domainScores"))
    domain_scores: dict[str, float] = {}
    if isinstance(domain_raw, dict):
        for k, v in domain_raw.items():
            key = str(k).strip().lower()
            if not key:
                continue
            domain_scores[key] = max(0.0, min(1.0, _to_float(v, 0.0)))

    return {
        "id": str(row.get("id", "") or "").strip(),
        "user_id": str(row.get("userId", safe_user_id) or safe_user_id).strip(),
        "inferred_careers": inferred_careers,
        "domain_scores": domain_scores,
        "computed_at": str(row.get("computed_at", "") or "").strip(),
    }