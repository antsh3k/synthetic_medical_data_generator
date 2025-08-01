# Synthetic Medical Data Generator
A simple tool to generate synthetic clinical data (patients, vitals, labs, encounters) by calling machine learning models hosted on Baseten. Designed for testing, development, and prototyping without using real patient information.

# 🧾 CLI Argument Categories

| Category               | Flag / Option                | Description                                                          |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------- |
| Diseases               | `--diseases`                 | Comma-separated list of target conditions (e.g. `"diabetes,asthma"`) |
| Number of patients     | `--patients` or `-n`         | Number of synthetic patients to generate                             |
| Documents per patient  | `--docs-range`               | Range as `min,max` (e.g., `1,5`)                                     |
| Time period            | `--start-date`, `--end-date` | Range of simulated data (e.g., `2022-01-01` to `2024-12-31`)         |
| Document type          | `--doc-types`                | Comma-separated types like `labs,vitals,encounters,notes`            |
| Output format          | `--output`                   | `json`, `csv`, or `stdout`                                           |
| Output file (optional) | `--out-path`                 | Where to save the file                                               |


# ✅ Example CLI Usage
```bash
python cli/generate.py \
  --diseases diabetes,hypertension \
  --patients 50 \
  --docs-range 2,6 \
  --start-date 2023-01-01 \
  --end-date 2025-01-01 \
  --doc-types labs,vitals \
  --output json \
  --out-path outputs/synthetic_data.json
```
