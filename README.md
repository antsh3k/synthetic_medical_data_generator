# Synthetic Medical Data Generator
An advanced tool to generate synthetic clinical data (patients, vitals, labs, encounters) using customizable templates and AI-powered medical validation. Designed for testing, development, and prototyping without using real patient information.

## 🎯 Enhanced Features

- 🏥 **Realistic Medical Data**: Generate clinically plausible patient demographics, vital signs, lab values, and medical notes
- 📋 **Template-Based Content**: Use customizable YAML templates for consistent document structure with variable content
- 🤖 **AI-Powered Validation**: LLM-based medical accuracy validation to prevent unrealistic combinations (e.g., male pregnancy)
- 🎛️ **Advanced Randomization**: Configurable randomization levels (conservative, moderate, high) with medical constraints
- 🔍 **Medical Consistency Checks**: Gender/age-appropriate conditions, realistic lab ranges, drug interactions
- 📊 **Multiple Output Formats**: Export as JSON, CSV, or view directly in stdout with validation scores
- ⚡ **Scalable Generation**: Batch processing with intelligent patient profiling and condition correlations
- 🎲 **Reproducible Results**: Seed-based generation for consistent testing scenarios

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

### Template-Based Generation (Recommended)
Generate 10 diabetes patients with specific lab templates:
```bash
python cli/generate.py --diseases diabetes --patients 10 --templates cardiology/labs/comprehensive_metabolic_panel --output json
```

### Advanced Generation with Validation
```bash
python cli/generate.py \
  --diseases diabetes,hypertension \
  --patients 25 \
  --templates cardiology/labs/comprehensive_metabolic_panel,cardiology/vitals/blood_pressure_monitoring \
  --randomization-level moderate \
  --medical-validation \
  --validation-strictness standard \
  --output json --out-path outputs/validated_data.json
```

### Legacy Mode (Backward Compatible)
```bash
python cli/generate.py --diseases diabetes --patients 10 --doc-types labs --output json
```

## 🧾 Enhanced CLI Arguments

### Core Generation Parameters
| Flag / Option          | Description                                      | Example Values                                  |
|------------------------|--------------------------------------------------|-------------------------------------------------|
| `--diseases`           | Comma-separated list of target conditions        | `"diabetes,asthma,hypertension,copd"`           |
| `--patients`, `-n`     | Number of synthetic patients to generate         | `10`, `100`, `1000`                             |
| `--docs-range`         | Range as `min,max` for documents per patient     | `1,5`, `2,8`, `3,10`                             |
| `--start-date`         | Start date for simulated encounters              | `2022-01-01`, `2024-12-31`                       |
| `--end-date`           | End date for simulated encounters                | `2022-01-01`, `2024-12-31`                       |

### Template System (NEW!)
| Flag / Option          | Description                                      | Example Values                                  |
|------------------------|--------------------------------------------------|-------------------------------------------------|
| `--templates`          | Comma-separated list of template paths          | `cardiology/labs/comprehensive_metabolic_panel` |
| `--template-dir`       | Directory containing custom templates            | `custom_templates`, `templates`                 |
| `--doc-types`          | **[LEGACY]** Document types (for compatibility) | `labs,vitals,encounters,notes,letters`         |

### Randomization Controls (NEW!)
| Flag / Option              | Description                                 | Example Values                          |
|----------------------------|---------------------------------------------|-----------------------------------------|
| `--randomization-level`    | Level of randomization in content          | `conservative`, `moderate`, `high`      |
| `--seed`                   | Random seed for reproducible results       | `42`, `12345`                          |

### Medical Validation (NEW!)
| Flag / Option              | Description                                 | Example Values                          |
|----------------------------|---------------------------------------------|-----------------------------------------|
| `--medical-validation`     | Enable LLM-based medical accuracy validation | (flag, no value)                       |
| `--validation-strictness`  | Level of validation strictness              | `basic`, `standard`, `strict`          |
| `--consistency-checks`     | Enable gender/age/condition consistency    | (flag, no value)                       |

### Output Options
| Flag / Option          | Description                                      | Example Values                                  |
|------------------------|--------------------------------------------------|-------------------------------------------------|
| `--output`             | Format for generated data                        | `json`, `csv`, `stdout`                          |
| `--out-path`           | File path for saved output (optional)            | `data/output.json`, `results.csv`               |

### Configuration
| Flag / Option          | Description                                      | Example Values                                  |
|------------------------|--------------------------------------------------|-------------------------------------------------|
| `--config`             | Path to custom configuration file                | `configs/cardiology.yaml`                        |
| `--api-key`            | Baseten API key (overrides `.env`)              | From Baseten dashboard                          |
| `--model-id`           | Baseten model ID (overrides `.env`)             | From Baseten dashboard                          |



## ✅ Enhanced Example Usage

### 1. Basic Template Generation
```bash
python cli/generate.py \
  --diseases diabetes \
  --patients 10 \
  --templates cardiology/labs/comprehensive_metabolic_panel \
  --output stdout
```

### 2. Advanced Multi-Template Generation with Validation
```bash
python cli/generate.py \
  --diseases diabetes,hypertension \
  --patients 50 \
  --docs-range 2,4 \
  --templates cardiology/labs/comprehensive_metabolic_panel,cardiology/vitals/blood_pressure_monitoring,endocrinology/labs/hba1c_diabetes_monitoring \
  --randomization-level moderate \
  --medical-validation \
  --validation-strictness standard \
  --consistency-checks \
  --output json \
  --out-path outputs/validated_medical_data.json \
  --seed 12345
```

### 3. High Randomization Research Mode
```bash
python cli/generate.py \
  --diseases diabetes,hypertension,heart_disease \
  --patients 100 \
  --docs-range 3,8 \
  --randomization-level high \
  --medical-validation \
  --validation-strictness strict \
  --output csv \
  --out-path research/high_variance_dataset.csv
```

### 4. Conservative Generation for Training Data
```bash
python cli/generate.py \
  --diseases diabetes \
  --patients 200 \
  --templates endocrinology/labs/hba1c_diabetes_monitoring \
  --randomization-level conservative \
  --medical-validation \
  --consistency-checks \
  --output json \
  --out-path training/clean_diabetes_data.json
```

## 📋 Template System

### Available Templates
The system includes pre-built templates organized by medical specialty:

```
templates/
├── cardiology/
│   ├── labs/
│   │   └── comprehensive_metabolic_panel.yaml
│   └── vitals/
│       └── blood_pressure_monitoring.yaml
├── endocrinology/
│   └── labs/
│       └── hba1c_diabetes_monitoring.yaml
└── general/
    ├── labs/
    ├── vitals/
    └── encounters/
```

### Template Features
- **Structured Content**: YAML-based templates with placeholders and randomization rules
- **Medical Constraints**: Age/gender appropriateness, condition-specific ranges
- **Validation Rules**: Built-in medical accuracy checks
- **Randomization Control**: Field-specific randomization parameters
- **Calculated Fields**: Automatic calculations (e.g., BMI, estimated average glucose)

### Creating Custom Templates
1. Create a new YAML file in the appropriate specialty directory
2. Define template structure with placeholders (`{{placeholder_name}}`)
3. Specify randomization parameters for each field
4. Add validation rules for medical accuracy
5. Include medical constraints (age, gender, conditions)

Example template structure:
```yaml
name: "Custom Lab Panel"
description: "Description of the test"
document_type: "labs"
specialty: "cardiology"

template:
  test_name: "{{test_name}}"
  patient_id: "{{patient_id}}"
  results:
    glucose:
      value: "{{glucose_value}}"
      unit: "mg/dL"
      randomization:
        distribution: "normal"
        mean: 95
        std: 15
        disease_modifiers:
          diabetes:
            mean: 180
            std: 40

validation_rules:
  - rule: "glucose_diabetes_check"
    condition: "diabetes in patient.conditions"
    validation: "glucose_value >= 126"
    severity: "warning"
```

## 🔍 Medical Validation

### Validation Features
- **Gender Consistency**: Prevents impossible combinations (e.g., male pregnancy)
- **Age Appropriateness**: Ensures age-appropriate conditions and values
- **Medical Accuracy**: Validates lab values against physiological ranges
- **Condition Correlation**: Checks for realistic disease-symptom relationships
- **Drug Interactions**: Basic medication compatibility checks

### Validation Levels
- **Basic**: Essential checks (gender/age consistency)
- **Standard**: Medical accuracy + basic checks (recommended)
- **Strict**: Comprehensive validation including drug interactions

### Understanding Validation Output
- ✅ **Valid**: Document passes all validation checks
- ⚠️ **Warnings**: Minor issues, usable with caution
- ❌ **Invalid**: Critical issues, should not be used
- 📊 **Scores**: Overall and medical accuracy scores (0-100)
