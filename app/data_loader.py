from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

from app.supabase_store import fetch_schools


class DataBundle:
    def __init__(self, schools: dict[str, dict], transcripts: list[dict], policy: dict, source: str = "unknown"):
        self.schools = schools
        self.transcripts = transcripts
        self.policy = policy
        self.source = source


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_bool(value: Any) -> bool:
    text = _safe_str(value).lower()
    return text in {"true", "1", "yes", "oui", "y", "vrai"}


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = _safe_str(value)
    if not text:
        return default
    text = text.replace(" ", "").replace(",", ".")
    try:
        return int(float(text))
    except ValueError:
        return default


def _to_iso_date(value: Any) -> str:
    if value is None:
        return "2026-01-01"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = _safe_str(value)
    if not text:
        return "2026-01-01"
    return text


def _normalize_token(value: Any) -> str:
    text = _safe_str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "general"


def _split_program_values(value: Any) -> list[str]:
    text = _safe_str(value)
    if not text:
        return []
    parts = re.split(r"[|,;/]+", text)
    out: list[str] = []
    for part in parts:
        token = _normalize_token(part)
        if token and token not in out:
            out.append(token)
    return out


def _split_program_labels(value: Any) -> list[str]:
    text = _safe_str(value)
    if not text:
        return []
    parts = re.split(r"[|,;/]+", text)
    labels: list[str] = []
    for part in parts:
        clean = " ".join(_safe_str(part).split())
        if clean and clean not in labels:
            labels.append(clean)
    return labels


def _legal_status_to_type(legal_status: str) -> str:
    s = _safe_str(legal_status).lower()
    return "public" if any(k in s for k in ["public", "publique", "etat", "state"]) else "private"


def load_from_supabase_schools(limit: int = 500) -> tuple[dict[str, dict], list[dict]]:
    response = fetch_schools(limit=limit)
    rows = response.get("items", []) if isinstance(response, dict) else []
    if not isinstance(rows, list) or not rows:
        return {}, []

    schools: dict[str, dict] = {}
    transcripts: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        raw_id = _safe_str(row.get("id"))
        if not raw_id:
            continue

        name = _safe_str(row.get("name")) or f"school_{idx}"
        city = _safe_str(row.get("city"))
        school_id = f"sb_{_normalize_token(raw_id)}"
        school_type = _legal_status_to_type(_safe_str(row.get("legal_status")))

        pricing_min = _parse_int(row.get("pricing_min"), default=-1)
        pricing_max = _parse_int(row.get("pricing_max"), default=-1)
        if pricing_min < 0 and pricing_max < 0:
            pricing_min = 0 if school_type == "public" else 15000
            pricing_max = 12000 if school_type == "public" else 50000
        elif pricing_min < 0:
            pricing_min = max(0, pricing_max)
        elif pricing_max < 0:
            pricing_max = max(0, pricing_min)

        programs = _split_program_values(row.get("programs_tags"))
        filieres = _split_program_values(row.get("filieres"))
        all_programs = sorted(set(programs + filieres)) or ["general"]
        legal_status = _safe_str(row.get("legal_status"))

        pretty_programs = _split_program_labels(row.get("programs_tags"))
        pretty_filieres = _split_program_labels(row.get("filieres"))
        pretty_program_labels = pretty_programs + [x for x in pretty_filieres if x not in pretty_programs]

        schools[school_id] = {
            "school_id": school_id,
            "name": name,
            "country": "MA",
            "city": city,
            "type": school_type,
            "legal_status": legal_status,
            "tuition_min_mad": pricing_min,
            "tuition_max_mad": pricing_max,
            "pricing_min": pricing_min,
            "pricing_max": pricing_max,
            "pricing_details": row.get("pricing_details"),
            "programs": all_programs,
            "programs_tags": _safe_str(row.get("programs_tags")),
            "filieres": _safe_str(row.get("filieres")),
            "admission_selectivity": "medium",
            "employability_score": 3.8,
            "salary_entry_min_mad": 6000,
            "salary_entry_max_mad": 12000,
            "international_double_degree": False,
            "website_url": _safe_str(row.get("website_url")),
            "logo_url": _safe_str(row.get("logo_url")),
            "acronym": _safe_str(row.get("acronym")),
            "source": "supabase_schools",
            "source_row_id": raw_id,
        }

        pricing_details = row.get("pricing_details")
        pricing_text = ""
        if isinstance(pricing_details, dict):
            pricing_text = json.dumps(pricing_details, ensure_ascii=True)
        elif pricing_details is not None:
            pricing_text = _safe_str(pricing_details)

        intro = name
        acronym = _safe_str(row.get("acronym"))
        if acronym:
            intro += f" ({acronym})"
        if city:
            intro += f" in {city}"

        overview_parts = [intro + "."]
        if legal_status:
            overview_parts.append(f"Status {legal_status}.")

        if pretty_program_labels:
            overview_parts.append(f"Programs: {' | '.join(pretty_program_labels[:14])}.")
        else:
            overview_parts.append(f"Programs: {' | '.join(all_programs[:14])}.")

        if pricing_min > 0 or pricing_max > 0:
            overview_parts.append(f"Tuition range {pricing_min} to {pricing_max} MAD.")

        overview_text = " ".join(overview_parts)
        if pricing_text:
            overview_text += f" Pricing details: {pricing_text}."

        transcripts.append(
            {
                "chunk_id": f"sb_{_normalize_token(raw_id)}_overview",
                "video_id": "supabase_schools",
                "school_id": school_id,
                "program": all_programs[0] if all_programs else "general",
                "level": "bac_plus_1",
                "language": "fr",
                "recorded_at": _safe_str(row.get("created_at")) or "2026-01-01",
                "text": overview_text,
                "sentiment": "positive",
                "tags": [
                    _normalize_token(city),
                    _normalize_token(school_type),
                    _normalize_token(_safe_str(row.get("acronym"))),
                ],
            }
        )

    return schools, transcripts


def _sheet_rows_by_headers(xlsx_path: Path) -> dict[str, list[dict[str, Any]]]:
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    tables: dict[str, list[dict[str, Any]]] = {}
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
        if not header_row:
            continue
        headers = [_safe_str(h) for h in header_row]
        key = "|".join(headers)
        rows: list[dict[str, Any]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            item = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
            if any(v is not None and _safe_str(v) for v in item.values()):
                rows.append(item)
        if rows:
            tables[key] = rows
    return tables


def load_from_excel_mcd(xlsx_path: Path) -> tuple[dict[str, dict], list[dict]]:
    tables = _sheet_rows_by_headers(xlsx_path)

    etab: list[dict[str, Any]] = []
    filiere: list[dict[str, Any]] = []
    cout: list[dict[str, Any]] = []
    candidature: list[dict[str, Any]] = []
    metier: list[dict[str, Any]] = []
    mener: list[dict[str, Any]] = []

    for key, rows in tables.items():
        cols = set(key.split("|"))
        if {"id_etablissement", "nom_etablissement"}.issubset(cols):
            etab = rows
        elif {"id_filiere", "id_etablissement", "nom_filiere"}.issubset(cols):
            filiere = rows
        elif {"id_cout", "id_filiere", "frais_scolarite_annuel"}.issubset(cols):
            cout = rows
        elif {"id_candidature", "id_filiere", "procedure"}.issubset(cols):
            candidature = rows
        elif {"id_metier", "nom_metier", "salaire_debutant_maroc"}.issubset(cols):
            metier = rows
        elif {"id_filiere", "id_metier"}.issubset(cols):
            mener = rows

    etab_by_id = {_safe_str(r.get("id_etablissement")): r for r in etab}
    filiere_by_id = {_safe_str(r.get("id_filiere")): r for r in filiere}

    filieres_by_etab: dict[str, list[dict[str, Any]]] = {}
    for row in filiere:
        etab_id = _safe_str(row.get("id_etablissement"))
        filieres_by_etab.setdefault(etab_id, []).append(row)

    cout_by_filiere: dict[str, list[dict[str, Any]]] = {}
    for row in cout:
        filiere_id = _safe_str(row.get("id_filiere"))
        cout_by_filiere.setdefault(filiere_id, []).append(row)

    candid_by_filiere: dict[str, list[dict[str, Any]]] = {}
    for row in candidature:
        filiere_id = _safe_str(row.get("id_filiere"))
        candid_by_filiere.setdefault(filiere_id, []).append(row)

    metier_by_id = {_safe_str(r.get("id_metier")): r for r in metier}
    metiers_by_filiere: dict[str, list[dict[str, Any]]] = {}
    for row in mener:
        filiere_id = _safe_str(row.get("id_filiere"))
        metier_id = _safe_str(row.get("id_metier"))
        linked = metier_by_id.get(metier_id)
        if linked:
            metiers_by_filiere.setdefault(filiere_id, []).append(linked)

    schools: dict[str, dict] = {}
    transcripts: list[dict] = []

    for etab_id, e in etab_by_id.items():
        school_id = f"uni_{_normalize_token(etab_id)}"
        filieres = filieres_by_etab.get(etab_id, [])

        programs = sorted(
            {
                _normalize_token(f.get("domaine") or f.get("nom_filiere"))
                for f in filieres
                if _safe_str(f.get("domaine") or f.get("nom_filiere"))
            }
        )
        domain_labels = sorted(
            {
                _safe_str(f.get("domaine"))
                for f in filieres
                if _safe_str(f.get("domaine"))
            }
        )
        filiere_labels = sorted(
            {
                _safe_str(f.get("nom_filiere"))
                for f in filieres
                if _safe_str(f.get("nom_filiere"))
            }
        )

        tuition_values: list[int] = []
        salary_values: list[int] = []
        employability_values: list[float] = []
        has_concours = False

        for f in filieres:
            filiere_id = _safe_str(f.get("id_filiere"))
            for c in cout_by_filiere.get(filiere_id, []):
                tuition = _parse_int(c.get("frais_scolarite_annuel"), default=-1)
                if tuition >= 0:
                    tuition_values.append(tuition)

            for m in metiers_by_filiere.get(filiere_id, []):
                salary = _parse_int(m.get("salaire_debutant_maroc"), default=-1)
                if salary >= 0:
                    salary_values.append(salary)
                emp = _safe_str(m.get("employabilite")).lower()
                emp_score = {"tres_haute": 4.8, "haute": 4.4, "moyenne": 3.5, "faible": 2.6}.get(emp)
                if emp_score is not None:
                    employability_values.append(emp_score)

            concours_text = _safe_str(f.get("concours_ou_dossier")).lower()
            if "concours" in concours_text:
                has_concours = True

        school_type_raw = _safe_str(e.get("type")).lower()
        school_type = "public" if "public" in school_type_raw else "private"
        tuition_min = min(tuition_values) if tuition_values else (0 if school_type == "public" else 15000)
        tuition_max = max(tuition_values) if tuition_values else (12000 if school_type == "public" else 50000)

        international = _parse_bool(e.get("partenariats_internationaux")) or (
            "double" in _safe_str(e.get("accreditations")).lower()
        )

        schools[school_id] = {
            "school_id": school_id,
            "name": _safe_str(e.get("nom_etablissement")) or school_id,
            "country": "MA",
            "city": _safe_str(e.get("ville")),
            "type": school_type,
            "tuition_min_mad": tuition_min,
            "tuition_max_mad": tuition_max,
            "programs": programs or ["general"],
            "programs_tags": " | ".join(domain_labels),
            "filieres": " | ".join(filiere_labels),
            "admission_selectivity": "high" if has_concours else "medium",
            "employability_score": round(mean(employability_values), 2) if employability_values else 3.8,
            "salary_entry_min_mad": min(salary_values) if salary_values else 6000,
            "salary_entry_max_mad": max(salary_values) if salary_values else 12000,
            "international_double_degree": international,
        }

    for filiere_id, f in filiere_by_id.items():
        etab_id = _safe_str(f.get("id_etablissement"))
        school_id = f"uni_{_normalize_token(etab_id)}"
        school = schools.get(school_id)
        if not school:
            continue

        c_row = (cout_by_filiere.get(filiere_id) or [{}])[0]
        cand_row = (candid_by_filiere.get(filiere_id) or [{}])[0]
        linked_jobs = metiers_by_filiere.get(filiere_id, [])

        metiers_text = ", ".join(_safe_str(m.get("nom_metier")) for m in linked_jobs[:4] if _safe_str(m.get("nom_metier")))
        bourse = "bourses disponibles" if _parse_bool(c_row.get("bourses_disponibles")) else "pas de bourse precisee"
        pay = "paiement echelonne possible" if _parse_bool(c_row.get("paiement_echelonne")) else "paiement standard"

        text = (
            f"Filiere {_safe_str(f.get('nom_filiere'))} ({_safe_str(f.get('diplome'))}), domaine {_safe_str(f.get('domaine'))}, "
            f"duree {_safe_str(f.get('duree_etudes'))} ans. Admission: {_safe_str(f.get('concours_ou_dossier'))}, "
            f"niveau acces {_safe_str(f.get('niveau_acces'))}, serie bac {_safe_str(f.get('serie_bac_requise'))}, "
            f"note minimale estimee {_safe_str(f.get('note_min_estimee'))}. Cout annuel estime {_safe_str(c_row.get('frais_scolarite_annuel'))} MAD, "
            f"{bourse}, {pay}. Debouches: {metiers_text or 'information en cours de consolidation'}."
        )

        niveau = _safe_str(f.get("niveau_acces")).lower()
        if "bac+3" in niveau or "bac+4" in niveau or "bac+5" in niveau:
            level = "bac_plus_3_plus"
        elif "bac+2" in niveau:
            level = "bac_plus_2"
        elif "bac+1" in niveau:
            level = "bac_plus_1"
        else:
            level = "bac"

        language_raw = _safe_str(etab_by_id.get(etab_id, {}).get("langue_enseignement")).lower()
        if "english" in language_raw or "anglais" in language_raw or language_raw == "en":
            language = "en"
        elif "ar" in language_raw:
            language = "ar"
        elif "dar" in language_raw:
            language = "darija"
        else:
            language = "fr"

        transcripts.append(
            {
                "chunk_id": f"xlsx_{_normalize_token(filiere_id)}",
                "video_id": "xlsx_mcd",
                "school_id": school_id,
                "program": _safe_str(f.get("nom_filiere")),
                "level": level,
                "language": language,
                "recorded_at": _to_iso_date(cand_row.get("date_concours") or cand_row.get("date_limite")),
                "text": text,
                "sentiment": "positive",
                "tags": [
                    _normalize_token(f.get("domaine")),
                    _normalize_token(f.get("diplome")),
                    _normalize_token(school.get("city", "")),
                ],
            }
        )

    return schools, transcripts


def load_schools(csv_path: Path) -> dict[str, dict]:
    schools: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            school_id = row["school_id"].strip()
            raw_programs = _safe_str(row.get("programs"))
            programs = [p.strip().lower() for p in raw_programs.split("|") if p.strip()]
            row["programs"] = programs
            row["programs_tags"] = raw_programs
            row["filieres"] = raw_programs
            row["tuition_min_mad"] = int(row["tuition_min_mad"])
            row["tuition_max_mad"] = int(row["tuition_max_mad"])
            row["salary_entry_min_mad"] = int(row["salary_entry_min_mad"])
            row["salary_entry_max_mad"] = int(row["salary_entry_max_mad"])
            row["employability_score"] = float(row["employability_score"])
            row["international_double_degree"] = row["international_double_degree"].lower() == "true"
            schools[school_id] = row
    return schools


def load_transcripts(jsonl_path: Path) -> list[dict]:
    transcripts: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if len(item.get("text", "")) < 40:
                continue
            transcripts.append(item)
    return transcripts


def load_from_json_catalog(json_path: Path) -> tuple[dict[str, dict], list[dict]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("etablissements", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    schools: dict[str, dict] = {}
    transcripts: list[dict] = []

    chunk_size = 4
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        name = _safe_str(row.get("nom") or row.get("nom_complet") or row.get("name"))
        city = _safe_str(row.get("ville") or row.get("city"))
        if not name:
            continue

        school_id = f"uni_{_normalize_token(name)}_{_normalize_token(city)}"
        filieres = row.get("filieres", [])
        if isinstance(filieres, list):
            filieres_list = [_safe_str(v) for v in filieres if _safe_str(v)]
        else:
            filieres_list = _split_program_labels(filieres)

        formations_list = _split_program_labels(row.get("formations"))
        domain_label = _safe_str(row.get("domaine"))

        program_labels = []
        if domain_label:
            program_labels.append(domain_label)
        program_labels.extend(filieres_list)
        for label in formations_list:
            if label not in program_labels:
                program_labels.append(label)

        programs = [_normalize_token(v) for v in program_labels if _safe_str(v)]
        programs = sorted(set(programs)) or ["general"]
        unique_labels: list[str] = []
        for label in program_labels:
            if label and label not in unique_labels:
                unique_labels.append(label)

        status_text = _safe_str(row.get("statut") or row.get("type") or row.get("legal_status")).lower()
        school_type = "public" if "public" in status_text else "private"

        frais = row.get("frais") if isinstance(row.get("frais"), dict) else {}
        frais_min = _parse_int(frais.get("min_MAD_an"), default=-1)
        frais_max = _parse_int(frais.get("max_MAD_an"), default=-1)

        annuels = _parse_int(row.get("frais_annuels_mad"), default=frais_max)
        inscription = _parse_int(row.get("frais_inscription_mad"), default=frais_min)
        if annuels < 0:
            annuels = 0 if school_type == "public" else 25000
        if inscription < 0:
            inscription = 0
        tuition_min = max(0, min(inscription, annuels))
        tuition_max = max(inscription, annuels)

        schools[school_id] = {
            "school_id": school_id,
            "name": name,
            "country": "MA",
            "city": city,
            "type": school_type,
            "tuition_min_mad": tuition_min,
            "tuition_max_mad": tuition_max,
            "programs": programs,
            "programs_tags": " | ".join(unique_labels),
            "filieres": " | ".join(filieres_list),
            "admission_selectivity": "medium",
            "employability_score": 3.8,
            "salary_entry_min_mad": 6000,
            "salary_entry_max_mad": 12000,
            "international_double_degree": False,
        }

        base_info = (
            f"{name} a {city}. Categorie {_safe_str(row.get('categorie') or row.get('sous_type') or row.get('type'))}. "
            f"Domaine {_safe_str(row.get('domaine'))}. Cycle {_safe_str(row.get('cycle') or row.get('diplomes'))}. "
            f"Langue {_safe_str(row.get('langue_enseignement') or 'fr')}."
        )
        admissions_info = (
            f"Admission: {_safe_str(row.get('conditions_acces') or row.get('acces'))}. "
            f"Type {_safe_str(row.get('type_etab') or row.get('sous_type') or row.get('type'))}. "
            f"Statut {_safe_str(row.get('statut') or row.get('type'))}."
        )
        cost_info = (
            f"Frais inscription {inscription} MAD, frais annuels {annuels} MAD. "
            f"Bourse: {_safe_str(row.get('bourse_disponible')) or 'non precisee'}. "
            f"Note frais: {_safe_str(row.get('frais_note') or frais.get('note'))}."
        )

        chunk_texts: list[tuple[str, str]] = [
            ("overview", base_info),
            ("admission", admissions_info),
            ("cost", cost_info),
        ]

        for j in range(0, len(filieres_list), chunk_size):
            subset = filieres_list[j : j + chunk_size]
            if not subset:
                continue
            prog_text = f"Filieres: {' | '.join(subset)}."
            chunk_texts.append(("programs", prog_text))

        for c_idx, (chunk_kind, chunk_text) in enumerate(chunk_texts, start=1):
            transcripts.append(
                {
                    "chunk_id": f"etab_{i + 1}_{chunk_kind}_{c_idx}",
                    "video_id": "json_catalog",
                    "school_id": school_id,
                    "program": programs[min(c_idx - 1, len(programs) - 1)] if programs else "general",
                    "level": "bac_plus_1",
                    "language": "fr",
                    "recorded_at": _safe_str(row.get("date_collecte")) or "2026-01-01",
                    "text": chunk_text,
                    "sentiment": "positive",
                    "tags": [
                        _normalize_token(row.get("categorie")),
                        _normalize_token(city),
                        _normalize_token(chunk_kind),
                    ],
                }
            )

    return schools, transcripts


def load_policy(policy_path: Path) -> dict:
    with policy_path.open("r", encoding="utf-8") as f:
        return {"raw_text": f.read()}


def load_bundle(root_dir: Path) -> DataBundle:
    strict_supabase = os.getenv("SUPABASE_STRICT_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
    # Explicit switch behavior:
    # - strict_supabase=True  -> DB-only mode (must load from Supabase)
    # - strict_supabase=False -> local documents/data mode
    if not strict_supabase:
        json_catalog_path = root_dir / "etablissements_maroc_complet.json"
        xlsx_candidates = [
            root_dir / "BDD_MCD_Universites.xlsx",
            root_dir / "etablissements_maroc_complet.xlsx",
        ]
        xlsx_path = next((p for p in xlsx_candidates if p.exists()), None)

        if json_catalog_path.exists():
            schools, transcripts = load_from_json_catalog(json_catalog_path)
        elif xlsx_path is not None:
            schools, transcripts = load_from_excel_mcd(xlsx_path)
        else:
            schools = {}
            transcripts = []

        policy = load_policy(root_dir / "config" / "policy_rules.yaml")
        return DataBundle(schools=schools, transcripts=transcripts, policy=policy, source="local_fallback")

    try:
        supabase_limit = _parse_int(os.getenv("SUPABASE_SCHOOLS_LIMIT", "500"), default=500)
        sb_schools, sb_transcripts = load_from_supabase_schools(limit=max(1, min(5000, supabase_limit)))
        if sb_schools:
            policy = load_policy(root_dir / "config" / "policy_rules.yaml")
            return DataBundle(schools=sb_schools, transcripts=sb_transcripts, policy=policy, source="supabase_schools")
    except Exception as exc:
        if strict_supabase:
            raise RuntimeError(f"Supabase schools load failed in strict mode: {exc}") from exc

    raise RuntimeError(
        "Supabase strict mode is enabled, but no schools were loaded from DB. "
        "Set SUPABASE_URL and SUPABASE_ANON_KEY, or switch to local mode with SUPABASE_STRICT_MODE=0."
    )
