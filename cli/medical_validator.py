"""
Medical Validation Service

Uses LLM models to validate synthetic medical data for accuracy,
consistency, and medical plausibility.
"""

import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Levels of validation strictness."""
    BASIC = "basic"
    STANDARD = "standard" 
    STRICT = "strict"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in medical data."""
    severity: ValidationSeverity
    category: str
    message: str
    field: str
    current_value: Any
    suggested_value: Optional[Any] = None
    rule_violated: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a medical document."""
    is_valid: bool
    overall_score: float  # 0-100
    issues: List[ValidationIssue]
    recommendations: List[str]
    patient_profile_consistent: bool
    medical_accuracy_score: float


class MedicalValidator:
    """Service for validating synthetic medical data using LLM models."""
    
    def __init__(self, api_key: str, model_id: str, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.api_key = api_key
        self.model_id = model_id
        self.validation_level = validation_level
        self.base_url = "https://model-{}.api.baseten.co".format(model_id)
        
        # Validation prompts for different types
        self.validation_prompts = {
            'patient_consistency': self._get_patient_consistency_prompt(),
            'medical_accuracy': self._get_medical_accuracy_prompt(),
            'value_ranges': self._get_value_ranges_prompt(),
            'gender_specific': self._get_gender_specific_prompt(),
            'age_appropriate': self._get_age_appropriate_prompt()
        }
    
    def validate_document(self, document: Dict[str, Any], patient_data: Dict[str, Any]) -> ValidationReport:
        """
        Validate a complete medical document for accuracy and consistency.
        
        Args:
            document: Generated medical document
            patient_data: Patient demographics and conditions
            
        Returns:
            ValidationReport with validation results
        """
        issues = []
        recommendations = []
        
        # Run different validation checks based on validation level
        validation_checks = self._get_validation_checks_for_level()
        
        for check_name in validation_checks:
            try:
                check_issues = self._run_validation_check(check_name, document, patient_data)
                issues.extend(check_issues)
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="validation_error",
                    message=f"Could not complete {check_name} validation: {str(e)}",
                    field="system"
                ))
        
        # Calculate overall scores
        overall_score = self._calculate_overall_score(issues)
        medical_accuracy_score = self._calculate_medical_accuracy_score(issues)
        patient_profile_consistent = self._assess_patient_consistency(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, document, patient_data)
        
        # Determine if overall validation passes
        critical_errors = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        
        is_valid = len(critical_errors) == 0 and len(errors) == 0
        
        return ValidationReport(
            is_valid=is_valid,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
            patient_profile_consistent=patient_profile_consistent,
            medical_accuracy_score=medical_accuracy_score
        )
    
    def _get_validation_checks_for_level(self) -> List[str]:
        """Get list of validation checks based on validation level."""
        base_checks = ['patient_consistency', 'gender_specific']
        
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            base_checks.extend(['medical_accuracy', 'value_ranges'])
        
        if self.validation_level == ValidationLevel.STRICT:
            base_checks.extend(['age_appropriate', 'drug_interactions', 'contraindications'])
        
        return base_checks
    
    def _run_validation_check(self, check_name: str, document: Dict[str, Any], 
                            patient_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Run a specific validation check using LLM."""
        prompt = self.validation_prompts.get(check_name)
        if not prompt:
            return []
        
        # Prepare the validation context
        context = {
            'patient': patient_data,
            'document': document,
            'document_type': document.get('_metadata', {}).get('template_path', 'unknown')
        }
        
        # Format the prompt with context
        formatted_prompt = prompt.format(**context)
        
        # Call LLM for validation
        try:
            response = self._call_baseten_llm(formatted_prompt)
            return self._parse_validation_response(response, check_name)
        except Exception as e:
            return [ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="llm_error",
                message=f"LLM validation failed for {check_name}: {str(e)}",
                field="system"
            )]
    
    def _call_baseten_llm(self, prompt: str) -> str:
        """Make API call to Baseten LLM model."""
        headers = {
            'Authorization': f'Api-Key {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'prompt': prompt,
            'max_tokens': 1000,
            'temperature': 0.1  # Low temperature for consistent validation
        }
        
        response = requests.post(
            f"{self.base_url}/production/predict",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Baseten API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result.get('data', result.get('output', ''))
    
    def _parse_validation_response(self, response: str, check_name: str) -> List[ValidationIssue]:
        """Parse LLM response into validation issues."""
        issues = []
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{') or response.strip().startswith('['):
                parsed = json.loads(response)
                if isinstance(parsed, list):
                    for issue_data in parsed:
                        issues.append(self._create_issue_from_dict(issue_data, check_name))
                elif isinstance(parsed, dict) and 'issues' in parsed:
                    for issue_data in parsed['issues']:
                        issues.append(self._create_issue_from_dict(issue_data, check_name))
            else:
                # Parse structured text response
                issues = self._parse_text_validation_response(response, check_name)
                
        except json.JSONDecodeError:
            # Fallback to text parsing
            issues = self._parse_text_validation_response(response, check_name)
        
        return issues
    
    def _create_issue_from_dict(self, issue_data: Dict[str, Any], check_name: str) -> ValidationIssue:
        """Create ValidationIssue from dictionary data."""
        return ValidationIssue(
            severity=ValidationSeverity(issue_data.get('severity', 'warning')),
            category=issue_data.get('category', check_name),
            message=issue_data.get('message', ''),
            field=issue_data.get('field', 'unknown'),
            current_value=issue_data.get('current_value'),
            suggested_value=issue_data.get('suggested_value'),
            rule_violated=issue_data.get('rule_violated')
        )
    
    def _parse_text_validation_response(self, response: str, check_name: str) -> List[ValidationIssue]:
        """Parse text-based validation response."""
        issues = []
        
        # Simple parsing logic - extend as needed
        lines = response.strip().split('\\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Look for issue indicators
            if any(keyword in line.lower() for keyword in ['error', 'invalid', 'incorrect', 'inconsistent']):
                severity = ValidationSeverity.ERROR
            elif any(keyword in line.lower() for keyword in ['warning', 'concern', 'unusual']):
                severity = ValidationSeverity.WARNING
            elif any(keyword in line.lower() for keyword in ['critical', 'dangerous', 'impossible']):
                severity = ValidationSeverity.CRITICAL
            else:
                severity = ValidationSeverity.INFO
            
            issues.append(ValidationIssue(
                severity=severity,
                category=check_name,
                message=line,
                field='parsed_text'
            ))
        
        return issues
    
    def _calculate_overall_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate overall validation score (0-100)."""
        if not issues:
            return 100.0
        
        # Weight different severity levels
        severity_weights = {
            ValidationSeverity.CRITICAL: -25,
            ValidationSeverity.ERROR: -10,
            ValidationSeverity.WARNING: -3,
            ValidationSeverity.INFO: -1
        }
        
        total_deduction = sum(severity_weights.get(issue.severity, 0) for issue in issues)
        score = max(0, 100 + total_deduction)
        
        return round(score, 1)
    
    def _calculate_medical_accuracy_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate medical accuracy specific score."""
        medical_issues = [i for i in issues if i.category in ['medical_accuracy', 'value_ranges', 'contraindications']]
        
        if not medical_issues:
            return 100.0
        
        # More aggressive scoring for medical accuracy
        severity_weights = {
            ValidationSeverity.CRITICAL: -30,
            ValidationSeverity.ERROR: -15,
            ValidationSeverity.WARNING: -5,
            ValidationSeverity.INFO: -1
        }
        
        total_deduction = sum(severity_weights.get(issue.severity, 0) for issue in medical_issues)
        score = max(0, 100 + total_deduction)
        
        return round(score, 1)
    
    def _assess_patient_consistency(self, issues: List[ValidationIssue]) -> bool:
        """Assess if document is consistent with patient profile."""
        consistency_issues = [i for i in issues if i.category in ['patient_consistency', 'gender_specific', 'age_appropriate']]
        critical_consistency_issues = [i for i in consistency_issues if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        
        return len(critical_consistency_issues) == 0
    
    def _generate_recommendations(self, issues: List[ValidationIssue], 
                                document: Dict[str, Any], 
                                patient_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        # Group issues by category
        issue_categories = {}
        for issue in issues:
            if issue.category not in issue_categories:
                issue_categories[issue.category] = []
            issue_categories[issue.category].append(issue)
        
        # Generate category-specific recommendations
        for category, category_issues in issue_categories.items():
            critical_issues = [i for i in category_issues if i.severity == ValidationSeverity.CRITICAL]
            error_issues = [i for i in category_issues if i.severity == ValidationSeverity.ERROR]
            
            if critical_issues:
                recommendations.append(f"URGENT: Address {len(critical_issues)} critical {category} issues immediately")
            elif error_issues:
                recommendations.append(f"Fix {len(error_issues)} {category} errors before using this data")
        
        # General recommendations
        if any(i.severity == ValidationSeverity.CRITICAL for i in issues):
            recommendations.append("Do not use this synthetic data until critical issues are resolved")
        
        return recommendations
    
    # Validation prompt templates
    def _get_patient_consistency_prompt(self) -> str:
        return """
        You are a medical expert validating synthetic medical data for consistency with patient demographics.
        
        Patient Profile:
        - Gender: {patient[gender]}
        - Age: {patient[age]}
        - Medical Conditions: {patient[conditions]}
        
        Medical Document:
        {document}
        
        Please analyze this medical document and identify any inconsistencies with the patient profile.
        Focus on:
        1. Gender-specific conditions or procedures
        2. Age-appropriate conditions and values
        3. Condition-specific findings consistency
        
        Return your findings as a JSON array of issues, each with:
        - severity: "critical", "error", "warning", or "info"
        - category: "patient_consistency"
        - message: Description of the issue
        - field: The specific field with the issue
        - current_value: The current value
        - suggested_value: Recommended correction (if applicable)
        
        If no issues found, return an empty array [].
        """
    
    def _get_medical_accuracy_prompt(self) -> str:
        return """
        You are a medical expert validating synthetic medical data for clinical accuracy.
        
        Patient: {patient[age]} year old {patient[gender]} with {patient[conditions]}
        
        Document Type: {document_type}
        Medical Data: {document}
        
        Please validate this medical data for:
        1. Realistic lab values and vital signs
        2. Appropriate reference ranges
        3. Logical correlations between values
        4. Medical plausibility
        
        Return findings as JSON array with severity, category, message, field, current_value, and suggested_value.
        Category should be "medical_accuracy".
        """
    
    def _get_value_ranges_prompt(self) -> str:
        return """
        You are validating lab values and vital signs for a {patient[age]} year old {patient[gender]}.
        
        Values to validate: {document}
        
        Check each numeric value against:
        1. Normal physiological ranges
        2. Critical value thresholds
        3. Age and gender-specific norms
        4. Condition-specific expected ranges
        
        Flag values that are:
        - Outside physiological possibility
        - Inconsistent with stated conditions
        - Would require immediate medical attention
        
        Return as JSON array with category "value_ranges".
        """
    
    def _get_gender_specific_prompt(self) -> str:
        return """
        Validate gender-specific medical appropriateness for {patient[gender]} patient.
        
        Medical data: {document}
        
        Check for:
        1. Gender-specific conditions (e.g., pregnancy in males)
        2. Gender-appropriate procedures
        3. Gender-specific normal ranges
        
        Return critical errors for impossible gender combinations.
        Category: "gender_specific"
        """
    
    def _get_age_appropriate_prompt(self) -> str:
        return """
        Validate age-appropriateness for {patient[age]} year old patient.
        
        Medical data: {document}
        
        Check for:
        1. Age-appropriate conditions
        2. Age-specific normal ranges
        3. Pediatric vs adult considerations
        
        Category: "age_appropriate"
        """


def validate_medical_data(document: Dict[str, Any], patient_data: Dict[str, Any],
                         api_key: str, model_id: str, 
                         validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
    """
    Convenience function to validate medical data.
    
    Args:
        document: Generated medical document
        patient_data: Patient demographics and conditions  
        api_key: Baseten API key
        model_id: Baseten model ID
        validation_level: Level of validation strictness
        
    Returns:
        ValidationReport with results
    """
    validator = MedicalValidator(api_key, model_id, validation_level)
    return validator.validate_document(document, patient_data)