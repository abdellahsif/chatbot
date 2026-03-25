from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


class DataBundle:
    def __init__(self, schools: dict[str, dict], transcripts: list[dict], policy: dict):
        self.schools = schools
        self.transcripts = transcripts
        self.policy = policy


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
            programs = [p.strip().lower() for p in row["programs"].split("|") if p.strip()]
            row["programs"] = programs
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


def load_policy(policy_path: Path) -> dict:
    with policy_path.open("r", encoding="utf-8") as f:
        return {"raw_text": f.read()}


def load_bundle(root_dir: Path) -> DataBundle:
    xlsx_path = root_dir / "BDD_MCD_Universites.xlsx"
    if xlsx_path.exists():
        schools, transcripts = load_from_excel_mcd(xlsx_path)
    else:
        schools = load_schools(root_dir / "data" / "mock" / "schools.csv")
        transcripts = load_transcripts(root_dir / "data" / "mock" / "transcripts.jsonl")
    policy = load_policy(root_dir / "config" / "policy_rules.yaml")
    return DataBundle(schools=schools, transcripts=transcripts, policy=policy)
