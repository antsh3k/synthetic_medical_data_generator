"""
Template Engine for Synthetic Medical Data Generator

Handles loading, processing, and rendering of medical document templates
with randomization and validation support.
"""

import os
import yaml
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass


@dataclass
class Patient:
    """Patient demographics and medical information."""
    id: str
    gender: str
    age: int
    conditions: List[str]
    medications: List[str] = None
    
    def __post_init__(self):
        if self.medications is None:
            self.medications = []


@dataclass 
class ValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    critical_issues: List[str]


class TemplateEngine:
    """Engine for processing medical document templates."""
    
    def __init__(self, template_dir: str = "templates", seed: Optional[int] = None):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.random = random.Random(seed) if seed else random.Random()
        self.np_random = np.random.RandomState(seed) if seed else np.random.RandomState()
        
        # Load available templates
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all available templates from the template directory."""
        if not self.template_dir.exists():
            print(f"Warning: Template directory '{self.template_dir}' does not exist")
            return
            
        for template_file in self.template_dir.rglob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                # Create template path (specialty/document_type/template_name)
                relative_path = template_file.relative_to(self.template_dir)
                template_path = str(relative_path.with_suffix(''))
                
                self.templates[template_path] = template_data
                
            except Exception as e:
                print(f"Warning: Could not load template {template_file}: {e}")
    
    def list_available_templates(self) -> List[str]:
        """Return list of all available template paths."""
        return list(self.templates.keys())
    
    def get_template(self, template_path: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by path."""
        return self.templates.get(template_path)
    
    def generate_document(self, template_path: str, patient: Patient, 
                         randomization_level: str = "moderate") -> Dict[str, Any]:
        """Generate a document from a template for a specific patient."""
        template = self.get_template(template_path)
        if not template:
            raise ValueError(f"Template not found: {template_path}")
        
        # Validate patient against template constraints
        validation = self._validate_patient_constraints(template, patient)
        if not validation.is_valid:
            raise ValueError(f"Patient validation failed: {validation.errors}")
        
        # Generate randomized values
        values = self._generate_randomized_values(template, patient, randomization_level)
        
        # Process template with values
        document = self._process_template(template, values)
        
        # Add metadata
        document['_metadata'] = {
            'template_path': template_path,
            'patient_id': patient.id,
            'generation_timestamp': datetime.now().isoformat(),
            'randomization_level': randomization_level
        }
        
        return document
    
    def _validate_patient_constraints(self, template: Dict[str, Any], 
                                    patient: Patient) -> ValidationResult:
        """Validate patient against template constraints."""
        errors = []
        warnings = []
        critical_issues = []
        
        constraints = template.get('constraints', {})
        
        # Check age range
        age_range = constraints.get('age_range')
        if age_range and not (age_range[0] <= patient.age <= age_range[1]):
            errors.append(f"Patient age {patient.age} outside template range {age_range}")
        
        # Check gender-specific constraints
        if constraints.get('gender_specific'):
            # Template-specific gender validations would go here
            pass
        
        # Check required conditions
        required_conditions = constraints.get('required_conditions', [])
        for condition in required_conditions:
            if condition not in patient.conditions:
                errors.append(f"Required condition '{condition}' not found in patient")
        
        # Check relevant conditions
        relevant_conditions = constraints.get('conditions_relevant', [])
        if relevant_conditions:
            has_relevant = any(cond in patient.conditions for cond in relevant_conditions)
            if not has_relevant:
                warnings.append(f"Patient has none of the relevant conditions: {relevant_conditions}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            critical_issues=critical_issues
        )
    
    def _generate_randomized_values(self, template: Dict[str, Any], 
                                  patient: Patient, 
                                  randomization_level: str) -> Dict[str, Any]:
        """Generate randomized values based on template specifications."""
        values = {}
        
        # Base randomization multiplier
        randomization_multipliers = {
            'conservative': 0.5,
            'moderate': 1.0,
            'high': 1.8
        }
        multiplier = randomization_multipliers.get(randomization_level, 1.0)
        
        # Process template structure recursively
        template_structure = template.get('template', {})
        values.update(self._generate_values_recursive(template_structure, patient, multiplier))
        
        # Add common placeholders
        values.update(self._generate_common_placeholders(patient))
        
        return values
    
    def _generate_values_recursive(self, structure: Union[Dict, List, Any], 
                                 patient: Patient, multiplier: float) -> Dict[str, Any]:
        """Recursively generate values from template structure."""
        values = {}
        
        if isinstance(structure, dict):
            for key, value in structure.items():
                if isinstance(value, dict) and 'randomization' in value:
                    # This is a field with randomization parameters
                    generated_value = self._generate_single_value(value, patient, multiplier)
                    values[key] = generated_value
                    
                    # Also create placeholder versions
                    if 'value' in value:
                        placeholder_key = value['value'].replace('{{', '').replace('}}', '')
                        values[placeholder_key] = generated_value
                        
                        # Add unit if present
                        if 'unit' in value:
                            values[f"{placeholder_key}_unit"] = value['unit']
                        if 'reference_range' in value:
                            values[f"{placeholder_key}_reference_range"] = value['reference_range']
                
                elif isinstance(value, (dict, list)):
                    values.update(self._generate_values_recursive(value, patient, multiplier))
        
        return values
    
    def _generate_single_value(self, field_spec: Dict[str, Any], 
                             patient: Patient, multiplier: float) -> float:
        """Generate a single randomized value based on field specification."""
        randomization = field_spec.get('randomization', {})
        
        # Base parameters
        distribution = randomization.get('distribution', 'normal')
        base_mean = randomization.get('mean', 0)
        base_std = randomization.get('std', 1)
        
        # Apply modifiers based on patient characteristics
        mean = base_mean
        std = base_std * multiplier
        
        # Gender modifiers
        gender_mods = randomization.get('gender_modifiers', {})
        if patient.gender.lower() in gender_mods:
            gender_mod = gender_mods[patient.gender.lower()]
            if 'mean' in gender_mod:
                mean = gender_mod['mean']
            if 'std' in gender_mod:
                std = gender_mod['std'] * multiplier
        
        # Age modifiers
        age_mods = randomization.get('age_modifiers', {})
        if patient.age >= 65 and 'elderly' in age_mods:
            age_mod = age_mods['elderly']
            if 'mean' in age_mod:
                mean = age_mod['mean']
            if 'std' in age_mod:
                std = age_mod['std'] * multiplier
        elif patient.age <= 30 and 'young' in age_mods:
            age_mod = age_mods['young']
            if 'mean' in age_mod:
                mean = age_mod['mean']
            if 'std' in age_mod:
                std = age_mod['std'] * multiplier
        
        # Disease modifiers
        disease_mods = randomization.get('disease_modifiers', {})
        for condition in patient.conditions:
            if condition in disease_mods:
                disease_mod = disease_mods[condition]
                if 'mean' in disease_mod:
                    mean = disease_mod['mean']
                if 'std' in disease_mod:
                    std = disease_mod['std'] * multiplier
                break  # Use first matching condition
        
        # Generate value based on distribution
        if distribution == 'normal':
            value = self.np_random.normal(mean, std)
        elif distribution == 'log_normal':
            # For log-normal, mean and std are for the underlying normal distribution
            value = self.np_random.lognormal(np.log(mean), std/mean)
        else:
            # Default to normal
            value = self.np_random.normal(mean, std)
        
        # Apply bounds if specified
        critical_values = field_spec.get('critical_values', {})
        if 'low' in critical_values:
            value = max(value, critical_values['low'])
        if 'high' in critical_values:
            value = min(value, critical_values['high'])
        
        # Round appropriately
        if isinstance(base_mean, int):
            value = int(round(value))
        else:
            value = round(value, 2)
        
        return value
    
    def _generate_common_placeholders(self, patient: Patient) -> Dict[str, Any]:
        """Generate common placeholder values."""
        now = datetime.now()
        
        # Generate patient name
        first_names_male = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Thomas"]
        first_names_female = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Wilson", "Moore"]
        
        if patient.gender.lower() == 'male':
            first_name = self.random.choice(first_names_male)
        else:
            first_name = self.random.choice(first_names_female)
        
        last_name = self.random.choice(last_names)
        patient_name = f"{first_name} {last_name}"
        
        # Generate birth date based on age
        birth_year = now.year - patient.age
        birth_month = self.random.randint(1, 12)
        birth_day = self.random.randint(1, 28)  # Safe day for all months
        patient_dob = f"{birth_month:02d}/{birth_day:02d}/{birth_year}"
        
        common_placeholders = {
            'patient_id': patient.id,
            'patient_name': patient_name,
            'patient_dob': patient_dob,
            'patient_mrn': f"MRN{self.random.randint(100000, 999999)}",
            'patient_phone': f"({self.random.randint(200, 999)}) {self.random.randint(200, 999)}-{self.random.randint(1000, 9999)}",
            
            'collection_date': (now - timedelta(days=self.random.randint(0, 30))).strftime('%Y-%m-%d'),
            'measurement_date': (now - timedelta(days=self.random.randint(0, 7))).strftime('%Y-%m-%d'),
            'measurement_time': f"{self.random.randint(8, 17):02d}:{self.random.randint(0, 59):02d}",
            'letter_date': now.strftime('%B %d, %Y'),
            'visit_date': (now - timedelta(days=self.random.randint(0, 14))).strftime('%B %d, %Y'),
            'signature_date': now.strftime('%B %d, %Y'),
            
            'physician_name': self.random.choice([
                "Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Jones",
                "Dr. Garcia", "Dr. Miller", "Dr. Davis", "Dr. Rodriguez", "Dr. Martinez"
            ]),
            'attending_physician': self.random.choice([
                "Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Davis", "Dr. Robert Wilson",
                "Dr. Lisa Rodriguez", "Dr. James Anderson", "Dr. Maria Garcia", "Dr. David Kim"
            ]),
            'physician_title': self.random.choice(["MD", "MD, PhD", "DO"]),
            'physician_specialty': self.random.choice([
                "Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology", "Pulmonology"
            ]),
            'provider_npi': f"{self.random.randint(1000000000, 9999999999)}",
            
            'referring_provider': self.random.choice([
                "Dr. Thompson", "Dr. Lee", "Dr. White", "Dr. Clark", "Dr. Lewis"
            ]),
            'provider_title': self.random.choice(["MD", "DO", "NP", "PA"]),
            'referring_practice': self.random.choice([
                "Primary Care Associates", "Family Health Center", "Community Medical Group",
                "Riverside Primary Care", "Downtown Family Practice"
            ]),
            'referring_address': self.random.choice([
                "123 Main St, Suite 200, Anytown, ST 12345",
                "456 Oak Ave, Medical Plaza, Anytown, ST 12345",
                "789 Elm St, Anytown, ST 12345"
            ]),
            
            'staff_name': self.random.choice([
                "Nurse Johnson", "Tech Williams", "RN Anderson", "MA Thompson", "LPN Garcia"
            ]),
            
            'clinic_name': self.random.choice([
                "Main Campus Clinic", "Downtown Medical Center", "Riverside Health",
                "University Hospital", "Community Care Center", "Metropolitan Medical Clinic"
            ]),
            'clinic_address': self.random.choice([
                "1000 Hospital Drive, Suite 300, Medical City, ST 12345",
                "555 Health Plaza, Anytown, ST 12345",
                "200 Wellness Way, Medical District, ST 12345"
            ]),
            'clinic_phone': f"({self.random.randint(200, 999)}) {self.random.randint(200, 999)}-{self.random.randint(1000, 9999)}",
            'clinic_fax': f"({self.random.randint(200, 999)}) {self.random.randint(200, 999)}-{self.random.randint(1000, 9999)}",
            
            'measurement_location': self.random.choice([
                "Exam Room 1", "Exam Room 2", "Clinic", "Emergency Department", "ICU"
            ]),
            
            'insurance_info': self.random.choice([
                "Blue Cross Blue Shield", "Aetna", "Cigna", "UnitedHealthcare", "Medicare"
            ]),
            
            'occupation': self.random.choice([
                "Teacher", "Engineer", "Nurse", "Accountant", "Retired", "Manager", "Sales", "Construction"
            ]),
            
            'exercise_habits': self.random.choice([
                "Walks 30 minutes daily", "Sedentary lifestyle", "Exercises 3x/week", "Active lifestyle", "Occasional walking"
            ]),
            
            'family_history': self.random.choice([
                "Father with diabetes, mother with hypertension",
                "No significant family history",
                "Mother with breast cancer, father with heart disease",
                "Diabetes and hypertension in family",
                "History of heart disease on paternal side"
            ])
        }
        
        # Add condition-specific placeholders
        if patient.conditions:
            primary_condition = patient.conditions[0]
            common_placeholders.update(self._generate_condition_specific_content(primary_condition, patient))
        
        return common_placeholders
    
    def _generate_condition_specific_content(self, condition: str, patient: Patient) -> Dict[str, Any]:
        """Generate condition-specific content for outpatient letters."""
        condition_content = {}
        
        if condition == 'diabetes':
            condition_content.update({
                'chief_complaint': 'Follow-up for diabetes management',
                'glucose_control_status': self.random.choice(['good', 'fair', 'poor']),
                'symptom_description': self.random.choice([
                    'Denies polyuria, polydipsia, or polyphagia',
                    'Reports occasional increased thirst',
                    'No concerning symptoms at this time',
                    'Some fatigue but otherwise stable'
                ]),
                'primary_diagnosis': 'Type 2 Diabetes Mellitus (E11.9)',
                'hpi_text': f'Patient with type 2 diabetes mellitus presents for routine follow-up. Reports {self.random.choice(["good", "fair", "suboptimal"])} glycemic control. {self.random.choice(["Denies polyuria, polydipsia, or polyphagia", "Reports occasional increased thirst", "Some fatigue but otherwise doing well"])}. Adherent to prescribed medications.'
            })
            
        elif condition == 'hypertension':
            condition_content.update({
                'chief_complaint': 'Follow-up for hypertension',
                'bp_control_status': self.random.choice(['well-controlled', 'moderately controlled', 'suboptimal']),
                'symptom_description': self.random.choice([
                    'Denies chest pain, shortness of breath, or headaches',
                    'Occasional mild headaches',
                    'No concerning cardiovascular symptoms'
                ]),
                'primary_diagnosis': 'Essential Hypertension (I10)',
                'hpi_text': f'Patient with essential hypertension presents for routine follow-up. Blood pressure has been {self.random.choice(["well-controlled", "moderately controlled", "elevated"])} with current regimen. {self.random.choice(["Denies chest pain or shortness of breath", "Reports adherence to medications", "No concerning symptoms"])}. Taking medications as prescribed.'
            })
            
        elif condition == 'asthma':
            condition_content.update({
                'chief_complaint': 'Follow-up for asthma management',
                'asthma_control_status': self.random.choice(['well-controlled', 'partially controlled', 'poorly controlled']),
                'rescue_frequency': self.random.choice(['rarely', '2-3 times per week', 'daily']),
                'symptom_description': self.random.choice([
                    'Denies wheezing or shortness of breath',
                    'Occasional mild wheezing with exertion',
                    'Some cough, especially at night'
                ]),
                'primary_diagnosis': 'Asthma, unspecified (J45.9)',
                'hpi_text': f'Patient with asthma presents for follow-up. Reports {self.random.choice(["good", "fair", "poor"])} control with current regimen. {self.random.choice(["Denies recent exacerbations", "Occasional mild symptoms", "Some nighttime awakening"])}. Using rescue inhaler {self.random.choice(["rarely", "2-3 times weekly", "daily"])}.'
            })
        
        # Add generic examination findings
        condition_content.update({
            'heent_exam': 'Normocephalic, atraumatic. PERRLA. No lymphadenopathy.',
            'cv_exam': self.random.choice([
                'Regular rate and rhythm. No murmurs, gallops, or rubs.',
                'RRR. Grade 2/6 systolic murmur at LUSB.',
                'Regular rate and rhythm. Normal S1, S2.'
            ]),
            'pulm_exam': self.random.choice([
                'Clear to auscultation bilaterally.',
                'Clear bilaterally with good air movement.',
                'Mild expiratory wheeze, otherwise clear.'
            ]),
            'abd_exam': 'Soft, non-tender, non-distended. Normal bowel sounds.',
            'neuro_exam': 'Alert and oriented x3. Grossly intact.',
            'ext_exam': 'No clubbing, cyanosis, or edema.',
            'skin_exam': 'Warm, dry, intact. No rashes or lesions.'
        })
        
        return condition_content
    
    def _process_template(self, template: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """Process template with generated values."""
        # For now, return the template structure with values injected
        # A full implementation would use a proper templating engine like Jinja2
        
        result = {}
        template_structure = template.get('template', {})
        
        # Simple placeholder replacement
        result = self._replace_placeholders_recursive(template_structure, values)
        
        # Add calculated fields
        calculated_fields = template.get('calculated_fields', {})
        for field_name, formula in calculated_fields.items():
            try:
                # Simple formula evaluation (in production, use a safer evaluator)
                result[field_name] = self._evaluate_formula(formula, values)
            except Exception as e:
                print(f"Warning: Could not calculate field {field_name}: {e}")
        
        return result
    
    def _replace_placeholders_recursive(self, obj: Any, values: Dict[str, Any]) -> Any:
        """Recursively replace placeholders in template structure."""
        if isinstance(obj, str):
            # Replace {{placeholder}} patterns
            def replace_placeholder(match):
                placeholder = match.group(1)
                return str(values.get(placeholder, match.group(0)))
            
            return re.sub(r'\{\{([^}]+)\}\}', replace_placeholder, obj)
        
        elif isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Skip special keys that aren't part of the output
                if key in ['randomization', 'critical_values', 'reference_range']:
                    continue
                result[key] = self._replace_placeholders_recursive(value, values)
            return result
        
        elif isinstance(obj, list):
            return [self._replace_placeholders_recursive(item, values) for item in obj]
        
        else:
            return obj
    
    def _evaluate_formula(self, formula: str, values: Dict[str, Any]) -> Any:
        """Safely evaluate a formula with values."""
        # This is a simplified implementation
        # In production, use a proper expression evaluator
        
        # Replace variable names with their values
        for var_name, var_value in values.items():
            if isinstance(var_value, (int, float)):
                formula = formula.replace(var_name, str(var_value))
        
        try:
            # Very basic evaluation - extend as needed
            return eval(formula)
        except:
            return None