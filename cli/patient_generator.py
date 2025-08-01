"""
Patient Generator for Synthetic Medical Data

Generates realistic patient profiles with demographics, conditions,
and medical history for use with medical document templates.
"""

import random
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from template_engine import Patient


class PatientGenerator:
    """Generates synthetic patient profiles with realistic medical characteristics."""
    
    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        
        # Medical condition data with demographics weighting
        self.conditions_data = {
            'diabetes': {
                'prevalence': 0.11,
                'age_weights': {(18, 30): 0.3, (31, 50): 1.0, (51, 70): 2.5, (71, 100): 3.0},
                'gender_weights': {'male': 1.1, 'female': 0.9},
                'related_conditions': ['hypertension', 'heart_disease', 'kidney_disease'],
                'common_medications': ['metformin', 'insulin', 'glipizide', 'empagliflozin']
            },
            'hypertension': {
                'prevalence': 0.45,
                'age_weights': {(18, 30): 0.1, (31, 50): 0.8, (51, 70): 2.0, (71, 100): 3.5},
                'gender_weights': {'male': 1.2, 'female': 0.8},
                'related_conditions': ['diabetes', 'heart_disease', 'stroke'],
                'common_medications': ['lisinopril', 'amlodipine', 'hydrochlorothiazide', 'metoprolol']
            },
            'asthma': {
                'prevalence': 0.08,
                'age_weights': {(18, 30): 1.5, (31, 50): 1.0, (51, 70): 0.8, (71, 100): 0.6},
                'gender_weights': {'male': 0.8, 'female': 1.2},
                'related_conditions': ['copd', 'allergies'],
                'common_medications': ['albuterol', 'fluticasone', 'montelukast', 'budesonide']
            },
            'copd': {
                'prevalence': 0.06,
                'age_weights': {(18, 30): 0.1, (31, 50): 0.3, (51, 70): 1.5, (71, 100): 3.0},
                'gender_weights': {'male': 1.1, 'female': 0.9},
                'related_conditions': ['asthma', 'heart_disease'],
                'common_medications': ['tiotropium', 'salmeterol', 'prednisone', 'oxygen']
            },
            'heart_disease': {
                'prevalence': 0.065,
                'age_weights': {(18, 30): 0.1, (31, 50): 0.5, (51, 70): 2.0, (71, 100): 4.0},
                'gender_weights': {'male': 1.4, 'female': 0.6},
                'related_conditions': ['diabetes', 'hypertension', 'stroke'],
                'common_medications': ['atorvastatin', 'clopidogrel', 'metoprolol', 'aspirin']
            },
            'obesity': {
                'prevalence': 0.36,
                'age_weights': {(18, 30): 0.8, (31, 50): 1.2, (51, 70): 1.3, (71, 100): 1.0},
                'gender_weights': {'male': 0.9, 'female': 1.1},
                'related_conditions': ['diabetes', 'hypertension', 'heart_disease'],
                'common_medications': ['orlistat', 'phentermine', 'liraglutide']
            },
            'colon_cancer': {
                'prevalence': 0.05,
                'age_weights': {(18, 30): 0.1, (31, 50): 0.5, (51, 70): 2.0, (71, 100): 3.5},
                'gender_weights': {'male': 1.1, 'female': 0.9},
                'related_conditions': ['obesity', 'diabetes'],
                'common_medications': ['5-fluorouracil', 'oxaliplatin', 'irinotecan', 'bevacizumab', 'cetuximab']
            }
        }
        
        # Common medication combinations
        self.medication_combinations = {
            ('diabetes', 'hypertension'): ['metformin', 'lisinopril'],
            ('diabetes', 'heart_disease'): ['metformin', 'atorvastatin', 'aspirin'],
            ('hypertension', 'heart_disease'): ['lisinopril', 'metoprolol', 'atorvastatin'],
            ('asthma', 'copd'): ['albuterol', 'tiotropium', 'fluticasone']
        }
    
    def generate_patients(self, count: int, target_diseases: Optional[List[str]] = None) -> List[Patient]:
        """
        Generate a list of synthetic patients.
        
        Args:
            count: Number of patients to generate
            target_diseases: Specific diseases to include (optional)
            
        Returns:
            List of Patient objects
        """
        patients = []
        
        for i in range(count):
            patient = self.generate_single_patient(target_diseases)
            patients.append(patient)
        
        return patients
    
    def generate_single_patient(self, target_diseases: Optional[List[str]] = None) -> Patient:
        """Generate a single synthetic patient."""
        
        # Generate basic demographics
        patient_id = f"P{str(uuid.uuid4())[:8].upper()}"
        gender = self.random.choice(['male', 'female'])
        age = self._generate_realistic_age()
        
        # Generate medical conditions
        conditions = self._generate_conditions(age, gender, target_diseases)
        
        # Generate medications based on conditions
        medications = self._generate_medications(conditions)
        
        return Patient(
            id=patient_id,
            gender=gender,
            age=age,
            conditions=conditions,
            medications=medications
        )
    
    def _generate_realistic_age(self) -> int:
        """Generate realistic age distribution for medical patients."""
        # Medical patients tend to skew older
        # Use weighted distribution
        
        age_ranges = [
            (18, 30, 0.15),  # Young adults - 15%
            (31, 50, 0.25),  # Middle age - 25%
            (51, 70, 0.40),  # Older adults - 40%
            (71, 90, 0.20)   # Elderly - 20%
        ]
        
        # Choose age range based on weights
        rand_val = self.random.random()
        cumulative = 0
        
        for min_age, max_age, weight in age_ranges:
            cumulative += weight
            if rand_val <= cumulative:
                return self.random.randint(min_age, max_age)
        
        # Fallback
        return self.random.randint(18, 85)
    
    def _generate_conditions(self, age: int, gender: str, 
                           target_diseases: Optional[List[str]] = None) -> List[str]:
        """Generate medical conditions based on demographics and targets."""
        conditions = []
        
        # If target diseases specified, include them with high probability
        if target_diseases:
            for disease in target_diseases:
                if disease in self.conditions_data:
                    # High probability for target diseases
                    if self.random.random() < 0.8:
                        conditions.append(disease)
        
        # Generate additional conditions based on prevalence and demographics
        for condition, data in self.conditions_data.items():
            if condition in conditions:
                continue  # Already added as target
            
            # Calculate probability based on demographics
            base_probability = data['prevalence']
            
            # Age weighting
            age_weight = 1.0
            for (min_age, max_age), weight in data['age_weights'].items():
                if min_age <= age <= max_age:
                    age_weight = weight
                    break
            
            # Gender weighting
            gender_weight = data['gender_weights'].get(gender, 1.0)
            
            # Condition interaction (comorbidities)
            interaction_weight = 1.0
            for existing_condition in conditions:
                if existing_condition in data.get('related_conditions', []):
                    interaction_weight *= 2.0  # Double probability for related conditions
            
            # Final probability
            final_probability = base_probability * age_weight * gender_weight * interaction_weight
            final_probability = min(final_probability, 0.9)  # Cap at 90%
            
            if self.random.random() < final_probability:
                conditions.append(condition)
        
        # Ensure at least one condition if target diseases specified
        if target_diseases and not conditions:
            conditions.append(self.random.choice(target_diseases))
        
        # If no conditions, add a common one based on age/gender
        if not conditions:
            if age > 50 and self.random.random() < 0.6:
                conditions.append('hypertension')
            elif age > 65 and self.random.random() < 0.4:
                conditions.append('diabetes')
        
        return conditions
    
    def _generate_medications(self, conditions: List[str]) -> List[str]:
        """Generate realistic medications based on conditions."""
        medications = []
        
        # Single condition medications
        for condition in conditions:
            if condition in self.conditions_data:
                condition_meds = self.conditions_data[condition]['common_medications']
                # Add 1-2 medications per condition
                num_meds = self.random.randint(1, min(2, len(condition_meds)))
                selected_meds = self.random.sample(condition_meds, num_meds)
                medications.extend(selected_meds)
        
        # Combination therapies
        condition_set = set(conditions)
        for combo_conditions, combo_meds in self.medication_combinations.items():
            if set(combo_conditions).issubset(condition_set):
                # Use combination therapy instead of individual meds
                medications = [m for m in medications if m not in combo_meds]
                medications.extend(combo_meds)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_medications = []
        for med in medications:
            if med not in seen:
                seen.add(med)
                unique_medications.append(med)
        
        return unique_medications
    
    def generate_patient_cohort(self, diseases: List[str], patients_per_disease: int) -> List[Patient]:
        """
        Generate a cohort of patients with specific disease distributions.
        
        Args:
            diseases: List of target diseases
            patients_per_disease: Number of patients per disease
            
        Returns:
            List of patients with distributed conditions
        """
        all_patients = []
        
        # Generate patients for each disease
        for disease in diseases:
            disease_patients = self.generate_patients(patients_per_disease, [disease])
            all_patients.extend(disease_patients)
        
        # Add some patients with multiple conditions
        multi_condition_count = len(diseases) * patients_per_disease // 4
        multi_condition_patients = self.generate_patients(multi_condition_count, diseases)
        all_patients.extend(multi_condition_patients)
        
        # Shuffle to mix patients
        self.random.shuffle(all_patients)
        
        return all_patients
    
    def get_patient_summary(self, patients: List[Patient]) -> Dict[str, Any]:
        """Generate summary statistics for a list of patients."""
        if not patients:
            return {}
        
        # Demographics
        total_patients = len(patients)
        gender_counts = {'male': 0, 'female': 0}
        age_sum = 0
        condition_counts = {}
        medication_counts = {}
        
        for patient in patients:
            gender_counts[patient.gender] += 1
            age_sum += patient.age
            
            for condition in patient.conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            for medication in patient.medications:
                medication_counts[medication] = medication_counts.get(medication, 0) + 1
        
        return {
            'total_patients': total_patients,
            'demographics': {
                'gender_distribution': {k: v/total_patients for k, v in gender_counts.items()},
                'average_age': round(age_sum / total_patients, 1),
                'age_range': [min(p.age for p in patients), max(p.age for p in patients)]
            },
            'conditions': {
                'most_common': sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                'total_unique_conditions': len(condition_counts),
                'avg_conditions_per_patient': round(sum(len(p.conditions) for p in patients) / total_patients, 1)
            },
            'medications': {
                'most_common': sorted(medication_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                'total_unique_medications': len(medication_counts),
                'avg_medications_per_patient': round(sum(len(p.medications) for p in patients) / total_patients, 1)
            }
        }