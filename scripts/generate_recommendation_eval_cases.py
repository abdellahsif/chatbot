import json
import random
from pathlib import Path

random.seed(42)

cities = [
    "Rabat",
    "Casablanca",
    "Marrakech",
    "Fes",
    "Tanger",
    "Agadir",
    "Oujda",
    "Kenitra",
    "Tetouan",
    "Meknes",
]

bac_streams = ["sm", "sm_a", "sm_b", "spc", "svt", "eco", "tgc", "lettres", "arts_appliques"]

grade_bands = ["passable", "assez_bien", "bien", "tres_bien", "excellent"]

motivations = ["employability", "prestige", "passion", "safety", "cash", "expat"]

budget_bands = ["zero_public", "tight_25k", "comfort_50k", "no_limit_70k_plus"]

domains = [
    "ai_data",
    "cybersecurity",
    "software",
    "business",
    "medicine",
    "architecture",
    "finance",
    "engineering",
    "design",
]

domain_to_career = {
    "ai_data": ["data scientist", "machine learning engineer"],
    "cybersecurity": ["cybersecurity analyst", "security engineer"],
    "software": ["software engineer", "full stack developer"],
    "business": ["business analyst", "management consultant"],
    "medicine": ["medical doctor", "pharmacist"],
    "architecture": ["architect", "interior architect"],
    "finance": ["financial analyst", "auditor"],
    "engineering": ["industrial engineer", "civil engineer"],
    "design": ["graphic designer", "ux designer"],
}

domain_scores = {
    "ai_data": {"computer": 0.9, "engineering": 0.7},
    "cybersecurity": {"computer": 0.9, "engineering": 0.6},
    "software": {"computer": 0.85, "engineering": 0.6},
    "business": {"business": 0.9, "finance": 0.6},
    "medicine": {"medicine": 0.95, "healthcare": 0.9},
    "architecture": {"engineering": 0.6, "arts": 0.7},
    "finance": {"finance": 0.95, "business": 0.6},
    "engineering": {"engineering": 0.95},
    "design": {"arts": 0.95},
}

ideal_schools_by_domain = {
    "ai_data": ["ENSIAS", "INPT", "UM6P"],
    "cybersecurity": ["INPT", "ENSIAS", "EHTP"],
    "software": ["ENSIAS", "EMI", "ENSA"],
    "business": ["ENCG", "ISCAE", "ESCA"],
    "medicine": ["FMP Rabat", "FMP Casablanca", "UM6SS"],
    "architecture": ["ENA Rabat", "EAC Casablanca"],
    "finance": ["ENCG", "ISCAE", "HEM"],
    "engineering": ["EMI", "EHTP", "ENSA"],
    "design": ["ESDI", "EAC Casablanca", "LISAA"],
}

unacceptable_by_domain = {
    "ai_data": ["FMP", "ENA"],
    "cybersecurity": ["FMP", "ENA"],
    "software": ["FMP", "ENA"],
    "business": ["FMP", "ENA"],
    "medicine": ["ENSA", "ENCG"],
    "architecture": ["FMP", "ENCG"],
    "finance": ["FMP", "ENA"],
    "engineering": ["FMP", "ENCG"],
    "design": ["FMP", "ENSA"],
}


def build_cases(count: int = 300) -> list[dict]:
    cases: list[dict] = []
    for i in range(1, count + 1):
        domain = random.choice(domains)
        city = random.choice(cities)
        budget = random.choice(budget_bands)
        bac = random.choice(bac_streams)
        grade = random.choice(grade_bands)
        motivation = random.choice(motivations)

        profile = {
            "bac_stream": bac,
            "expected_grade_band": grade,
            "motivation": motivation,
            "budget_band": budget,
            "city": city,
            "country": "MA",
        }

        career_profile = {
            "inferred_careers": domain_to_career[domain],
            "domain_scores": domain_scores[domain],
        }

        ideal = []
        for base in ideal_schools_by_domain[domain]:
            if base == "ENSA":
                ideal.append(f"ENSA {city}")
            else:
                ideal.append(base)

        unacceptable = list(unacceptable_by_domain[domain])
        if budget == "zero_public":
            unacceptable = list(set(unacceptable + ["ESCA", "HEM", "AUI", "UIR"]))

        cases.append(
            {
                "id": f"case_{i:03d}",
                "profile": profile,
                "career_profile": career_profile,
                "ideal_schools": ideal,
                "unacceptable_schools": unacceptable,
            }
        )
    return cases


if __name__ == "__main__":
    out_path = Path("eval") / "recommendation_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases(300)
    out_path.write_text(json.dumps(cases, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out_path}")
