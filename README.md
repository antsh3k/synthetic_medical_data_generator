# Synthetic Medical Data Generator
A simple tool to generate synthetic clinical data (patients, vitals, labs, encounters) by calling machine learning models hosted on Baseten. Designed for testing, development, and prototyping without using real patient information.

## 🎯 Features

- 🏥 Realistic Medical Data: Generate clinically plausible patient demographics, vital signs, lab values, and medical notes
- 🤖 AI-Powered Content: Uses LLM models via Baseten to create natural, contextual medical documentation
- 🔧 Highly Configurable: Customize diseases, patient counts, document types, and time ranges
- 📊 Multiple Output Formats: Export as JSON, CSV, or view directly in stdout
- ⚡ Scalable Generation: Batch processing with rate limiting and error handling
- 🎲 Reproducible Results: Seed-based generation for consistent testing scenarios

## 🚀 Quick Start
Installation
bash# Clone the repository
git clone <your-repo-url>
cd synthetic-medical-data-generator
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Baseten credentials
```

## Basic Usage
Generate 10 diabetes patients with lab data:
```bash
python cli/generate.py --diseases diabetes --patients 10 --doc-types labs --output json
```


## 🧾 CLI Argument Categories

| Category               | Flag / Option                   | Description                                      | Example Values                                  |
|------------------------|----------------------------------|--------------------------------------------------|-------------------------------------------------|
| Diseases               | `--diseases`                    | Comma-separated list of target conditions        | `"diabetes,asthma,hypertension,copd"`           |
| Number of patients     | `--patients`, `-n`              | Number of synthetic patients to generate         | `10`, `100`, `1000`                             |
| Documents per patient  | `--docs-range`                  | Range as `min,max` for documents per patient     | `1,5`, `2,8`, `3,10`                             |
| Time period            | `--start-date`, `--end-date`    | Date range for simulated encounters              | `2022-01-01`, `2024-12-31`                       |
| Document type          | `--doc-types`                   | Types of clinical documents to generate          | `labs`, `vitals`, `encounters`, `notes`, `letters` |
| Output format          | `--output`                      | Format for generated data                        | `json`, `csv`, `stdout`                          |
| Output file            | `--out-path`                    | File path for saved output (optional)            | `data/output.json`, `results.csv`               |
| Configuration          | `--config`                      | Path to custom configuration file                | `configs/cardiology.yaml`                        |
| Seed                   | `--seed`                        | Random seed for reproducible results             | `42`, `12345`                                   |
| API Settings           | `--api-key`, `--model-id`       | Baseten credentials (can override `.env`)        | From Baseten dashboard                          |



## ✅ Example CLI Usage
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
