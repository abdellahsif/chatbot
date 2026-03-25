# Data Schema (MVP)

## 1) Schools (`schools.csv`)
Required columns:
- `school_id` (string, unique)
- `name` (string)
- `country` (enum: MA, SN, CI)
- `city` (string)
- `type` (enum: public, private)
- `tuition_min_mad` (int)
- `tuition_max_mad` (int)
- `programs` (pipe-separated string, example: `computer_science|business`)
- `admission_selectivity` (enum: low, medium, high)
- `employability_score` (float 0-5)
- `salary_entry_min_mad` (int)
- `salary_entry_max_mad` (int)
- `international_double_degree` (bool)

## 2) Transcript Snippets (`transcripts.jsonl`)
Required fields per line:
- `chunk_id` (string, unique)
- `video_id` (string)
- `school_id` (string, FK to schools)
- `program` (string)
- `level` (enum: bac, bac_plus_1, bac_plus_2, bac_plus_3_plus)
- `language` (enum: fr, ar, darija, en)
- `recorded_at` (YYYY-MM-DD)
- `text` (string)
- `sentiment` (enum: positive, mixed, negative)
- `tags` (array of strings)

## 3) User Profile (runtime object)
Required keys:
- `bac_stream` (enum: sm, spc, svt, eco, lettres)
- `expected_grade_band` (enum: passable, bien, tres_bien, elite)
- `motivation` (enum: cash, prestige, passion, safety, expat)
- `budget_band` (enum: zero_public, tight_25k, comfort_50k, no_limit_70k_plus)
- `city` (string)
- `country` (enum: MA, SN, CI)

## Validation Rules
- Reject records without `school_id` in both datasets.
- Reject transcript lines where `text` length < 40 chars.
- Normalize strings to lowercase except `name` and `city` display fields.
- Convert all currencies to MAD in MVP.
