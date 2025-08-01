#!/usr/bin/env python3
"""
Synthetic Medical Data Generator CLI

Generates realistic clinical data using AI models and customizable templates.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with enhanced template and validation options."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical data using AI models and templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation with templates
  python cli/generate.py --templates cardiology/labs/basic_metabolic_panel --patients 10

  # Advanced generation with validation
  python cli/generate.py \\
    --diseases diabetes,hypertension \\
    --patients 50 \\
    --templates endocrinology/labs/hba1c,cardiology/vitals/blood_pressure \\
    --randomization-level moderate \\
    --medical-validation \\
    --validation-strictness standard \\
    --output json --out-path outputs/validated_data.json

  # Custom templates with high randomization
  python cli/generate.py \\
    --template-dir custom_templates \\
    --templates my_specialty/custom_template \\
    --randomization-level high \\
    --patients 25
        """
    )

    # Core generation parameters
    core_group = parser.add_argument_group('Core Generation Parameters')
    core_group.add_argument('--diseases', type=str, 
                           help='Comma-separated list of target conditions (diabetes,asthma,hypertension,copd)')
    core_group.add_argument('--patients', '-n', type=int, default=10,
                           help='Number of synthetic patients to generate (default: 10)')
    core_group.add_argument('--docs-range', type=str, default='1,3',
                           help='Range as min,max for documents per patient (default: 1,3)')
    core_group.add_argument('--start-date', type=str, default='2023-01-01',
                           help='Start date for simulated encounters (YYYY-MM-DD)')
    core_group.add_argument('--end-date', type=str, default='2024-12-31',
                           help='End date for simulated encounters (YYYY-MM-DD)')

    # Template system (NEW)
    template_group = parser.add_argument_group('Template System')
    template_group.add_argument('--templates', type=str,
                               help='Comma-separated list of templates (e.g., cardiology/labs/comprehensive_metabolic_panel)')
    template_group.add_argument('--template-dir', type=str, default='templates',
                               help='Directory containing custom templates (default: templates)')
    template_group.add_argument('--doc-types', type=str,
                               help='[LEGACY] Document types: labs,vitals,encounters,notes,letters')

    # Randomization controls (NEW)
    random_group = parser.add_argument_group('Randomization Controls')
    random_group.add_argument('--randomization-level', type=str, 
                             choices=['conservative', 'moderate', 'high'], default='moderate',
                             help='Level of randomization in generated content (default: moderate)')
    random_group.add_argument('--seed', type=int,
                             help='Random seed for reproducible results')

    # Medical validation (NEW)
    validation_group = parser.add_argument_group('Medical Validation')
    validation_group.add_argument('--medical-validation', action='store_true',
                                 help='Enable LLM-based medical accuracy validation')
    validation_group.add_argument('--validation-strictness', type=str,
                                 choices=['basic', 'standard', 'strict'], default='standard',
                                 help='Level of validation strictness (default: standard)')
    validation_group.add_argument('--consistency-checks', action='store_true',
                                 help='Enable gender/age/condition consistency validation')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str, choices=['json', 'csv', 'stdout'], 
                             default='stdout', help='Output format (default: stdout)')
    output_group.add_argument('--out-path', type=str,
                             help='File path for saved output (optional)')

    # Configuration and API
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str,
                             help='Path to custom configuration file')
    config_group.add_argument('--api-key', type=str,
                             help='Baseten API key (overrides .env)')
    config_group.add_argument('--model-id', type=str,
                             help='Baseten model ID (overrides .env)')

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate argument combinations and requirements."""
    # Ensure either templates or doc-types is specified
    if not args.templates and not args.doc_types:
        print("Error: Must specify either --templates or --doc-types", file=sys.stderr)
        sys.exit(1)
    
    # Validate date format
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Invalid date format. Use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    
    # Validate docs-range format
    try:
        min_docs, max_docs = map(int, args.docs_range.split(','))
        if min_docs < 1 or max_docs < min_docs:
            raise ValueError
    except ValueError:
        print("Error: Invalid docs-range format. Use min,max (e.g., 1,5)", file=sys.stderr)
        sys.exit(1)
    
    # Check if template directory exists
    if not os.path.exists(args.template_dir):
        print(f"Warning: Template directory '{args.template_dir}' does not exist")


def load_configuration(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    default_config = {
        'baseten': {
            'api_key': os.getenv('BASETEN_API_KEY'),
            'model_id': os.getenv('BASETEN_MODEL_ID')
        },
        'validation': {
            'enable_medical_validation': False,
            'strictness': 'standard',
            'consistency_checks': False
        },
        'randomization': {
            'level': 'moderate',
            'seed': None
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(file_config)
        except ImportError:
            print("Warning: PyYAML not installed. Cannot load config file.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load config file: {e}", file=sys.stderr)
    
    return default_config


def main():
    """Main entry point for the synthetic medical data generator."""
    args = parse_arguments()
    validate_arguments(args)
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Override config with CLI arguments
    if args.api_key:
        config['baseten']['api_key'] = args.api_key
    if args.model_id:
        config['baseten']['model_id'] = args.model_id
    if args.medical_validation:
        config['validation']['enable_medical_validation'] = True
        config['validation']['strictness'] = args.validation_strictness
    if args.consistency_checks:
        config['validation']['consistency_checks'] = True
    
    config['randomization']['level'] = args.randomization_level
    if args.seed:
        config['randomization']['seed'] = args.seed

    print("🏥 Synthetic Medical Data Generator")
    print("=" * 50)
    print(f"Patients to generate: {args.patients}")
    print(f"Documents per patient: {args.docs_range}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Randomization level: {args.randomization_level}")
    
    if args.templates:
        print(f"Templates: {args.templates}")
    if args.doc_types:
        print(f"Document types (legacy): {args.doc_types}")
    
    if config['validation']['enable_medical_validation']:
        print(f"Medical validation: Enabled ({config['validation']['strictness']})")
    
    print("\n🔄 Generating synthetic medical data...")
    
    try:
        # Generate synthetic medical data
        results = generate_synthetic_data(args, config)
        
        # Output results
        output_results(results, args)
        
        print(f"\n✅ Successfully generated data for {len(results['patients'])} patients")
        if results.get('validation_summary'):
            print(f"📊 Validation summary: {results['validation_summary']}")
        
    except Exception as e:
        print(f"\n❌ Error generating data: {str(e)}")
        sys.exit(1)


def generate_synthetic_data(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic medical data using the enhanced pipeline."""
    from patient_generator import PatientGenerator
    from template_engine import TemplateEngine
    from medical_validator import MedicalValidator, ValidationLevel
    
    # Initialize components
    patient_gen = PatientGenerator(seed=config['randomization']['seed'])
    template_engine = TemplateEngine(
        template_dir=args.template_dir,
        seed=config['randomization']['seed']
    )
    
    # Initialize validator if enabled
    validator = None
    if config['validation']['enable_medical_validation']:
        if not config['baseten']['api_key'] or not config['baseten']['model_id']:
            print("⚠️  Warning: Medical validation enabled but missing Baseten credentials")
        else:
            validation_level = ValidationLevel(config['validation']['strictness'])
            validator = MedicalValidator(
                config['baseten']['api_key'],
                config['baseten']['model_id'],
                validation_level
            )
    
    # Parse target diseases
    target_diseases = None
    if args.diseases:
        target_diseases = [d.strip() for d in args.diseases.split(',')]
    
    # Generate patients
    print(f"👥 Generating {args.patients} patients...")
    patients = patient_gen.generate_patients(args.patients, target_diseases)
    
    # Get patient summary
    patient_summary = patient_gen.get_patient_summary(patients)
    print(f"   Average age: {patient_summary['demographics']['average_age']} years")
    print(f"   Most common conditions: {', '.join([c[0] for c in patient_summary['conditions']['most_common'][:3]])}")
    
    # Determine templates to use
    templates_to_use = []
    if args.templates:
        templates_to_use = [t.strip() for t in args.templates.split(',')]
    elif args.doc_types:
        # Legacy mode - convert doc types to templates
        templates_to_use = convert_doc_types_to_templates(args.doc_types, template_engine)
    
    if not templates_to_use:
        # Default to generic outpatient clinic letter
        default_template = "general/letters/outpatient_clinic_letter"
        available_templates = template_engine.list_available_templates()
        
        if default_template in available_templates:
            templates_to_use = [default_template]
        elif available_templates:
            templates_to_use = available_templates[:3]  # Fallback to first 3 templates
        else:
            raise ValueError("No templates available and none specified")
    
    print(f"📄 Using templates: {', '.join(templates_to_use)}")
    
    # Parse documents per patient range
    min_docs, max_docs = map(int, args.docs_range.split(','))
    
    # Generate documents
    all_documents = []
    validation_results = []
    
    print(f"📝 Generating documents...")
    
    for i, patient in enumerate(patients):
        print(f"   Patient {i+1}/{len(patients)}: {patient.id}", end=" ")
        
        # Determine number of documents for this patient
        num_docs = patient_gen.random.randint(min_docs, max_docs)
        
        # Generate documents for this patient
        patient_documents = []
        
        for doc_num in range(num_docs):
            # Select template (rotate through available templates)
            template_path = templates_to_use[doc_num % len(templates_to_use)]
            
            try:
                # Generate document
                document = template_engine.generate_document(
                    template_path, 
                    patient, 
                    args.randomization_level
                )
                
                # Validate document if validation is enabled
                validation_report = None
                if validator:
                    try:
                        patient_data = {
                            'gender': patient.gender,
                            'age': patient.age,
                            'conditions': patient.conditions,
                            'medications': patient.medications
                        }
                        validation_report = validator.validate_document(document, patient_data)
                        validation_results.append(validation_report)
                        
                        # Skip document if critical validation errors
                        if not validation_report.is_valid and any(
                            issue.severity.value == 'critical' 
                            for issue in validation_report.issues
                        ):
                            print("❌", end="")
                            continue
                        elif validation_report.issues:
                            print("⚠️", end="")
                        else:
                            print("✅", end="")
                    except Exception as e:
                        print(f"⚠️ Validation error: {e}", end="")
                else:
                    print("✓", end="")
                
                # Add validation report to document metadata
                if validation_report:
                    document['_validation'] = {
                        'is_valid': validation_report.is_valid,
                        'overall_score': validation_report.overall_score,
                        'medical_accuracy_score': validation_report.medical_accuracy_score,
                        'issues_count': len(validation_report.issues)
                    }
                
                patient_documents.append(document)
                
            except Exception as e:
                print(f"❌ Error generating document: {e}", end="")
        
        print()  # New line after patient
        all_documents.extend(patient_documents)
    
    # Compile results
    results = {
        'patients': [
            {
                'id': p.id,
                'gender': p.gender,
                'age': p.age,
                'conditions': p.conditions,
                'medications': p.medications
            } for p in patients
        ],
        'documents': all_documents,
        'patient_summary': patient_summary,
        'generation_metadata': {
            'total_patients': len(patients),
            'total_documents': len(all_documents),
            'templates_used': templates_to_use,
            'randomization_level': args.randomization_level,
            'validation_enabled': config['validation']['enable_medical_validation'],
            'generation_timestamp': datetime.now().isoformat()
        }
    }
    
    # Add validation summary if validation was performed
    if validation_results:
        valid_docs = sum(1 for vr in validation_results if vr.is_valid)
        avg_score = sum(vr.overall_score for vr in validation_results) / len(validation_results)
        avg_medical_score = sum(vr.medical_accuracy_score for vr in validation_results) / len(validation_results)
        
        results['validation_summary'] = {
            'total_validated': len(validation_results),
            'valid_documents': valid_docs,
            'validation_rate': round(valid_docs / len(validation_results) * 100, 1),
            'average_overall_score': round(avg_score, 1),
            'average_medical_accuracy_score': round(avg_medical_score, 1)
        }
    
    return results


def convert_doc_types_to_templates(doc_types: str, template_engine: TemplateEngine) -> List[str]:
    """Convert legacy doc-types to template paths."""
    doc_type_list = [dt.strip() for dt in doc_types.split(',')]
    available_templates = template_engine.list_available_templates()
    
    # Simple mapping - find templates that contain the doc type
    templates = []
    for doc_type in doc_type_list:
        matching_templates = [t for t in available_templates if doc_type in t]
        if matching_templates:
            templates.extend(matching_templates[:2])  # Max 2 templates per doc type
    
    return templates


def output_results(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Output results in the specified format."""
    
    if args.output == 'stdout':
        # Pretty print to stdout
        print("\n" + "="*60)
        print("SYNTHETIC MEDICAL DATA RESULTS")
        print("="*60)
        
        for i, patient in enumerate(results['patients']):
            print(f"\nPATIENT {i+1}: {patient['id']}")
            print(f"Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"Conditions: {', '.join(patient['conditions'])}")
            if patient['medications']:
                print(f"Medications: {', '.join(patient['medications'])}")
            
            # Find documents for this patient
            patient_docs = [d for d in results['documents'] 
                          if d.get('_metadata', {}).get('patient_id') == patient['id']]
            
            for j, doc in enumerate(patient_docs):
                print(f"\n  DOCUMENT {j+1}: {doc.get('_metadata', {}).get('template_path', 'Unknown')}")
                if '_validation' in doc:
                    val = doc['_validation']
                    print(f"  Validation: {'✅ Valid' if val['is_valid'] else '❌ Invalid'} "
                          f"(Score: {val['overall_score']}/100)")
                
                # Print key document content (simplified)
                if 'results' in doc:
                    print("  Lab Results:")
                    for test_name, test_data in doc['results'].items():
                        if isinstance(test_data, dict) and 'value' in test_data:
                            unit = test_data.get('unit', '')
                            print(f"    {test_name}: {test_data['value']} {unit}")
                
                elif 'vital_signs' in doc:
                    print("  Vital Signs:")
                    for vital_name, vital_data in doc['vital_signs'].items():
                        if isinstance(vital_data, dict):
                            if 'systolic' in vital_data and 'diastolic' in vital_data:
                                print(f"    {vital_name}: {vital_data['systolic']['value']}/{vital_data['diastolic']['value']} mmHg")
                            elif 'value' in vital_data:
                                unit = vital_data.get('unit', '')
                                print(f"    {vital_name}: {vital_data['value']} {unit}")
    
    elif args.output in ['json', 'csv']:
        if args.out_path:
            output_path = args.out_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/synthetic_medical_data_{timestamp}.{args.output}"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if args.output == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {output_path}")
        
        elif args.output == 'csv':
            import csv
            # Flatten data for CSV export
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'patient_id', 'age', 'gender', 'conditions', 'medications',
                    'document_type', 'template_path', 'validation_score', 'generation_timestamp'
                ])
                
                # Write data rows
                for doc in results['documents']:
                    metadata = doc.get('_metadata', {})
                    patient_id = metadata.get('patient_id', '')
                    
                    # Find patient data
                    patient = next((p for p in results['patients'] if p['id'] == patient_id), {})
                    
                    validation_score = doc.get('_validation', {}).get('overall_score', '')
                    
                    writer.writerow([
                        patient_id,
                        patient.get('age', ''),
                        patient.get('gender', ''),
                        '; '.join(patient.get('conditions', [])),
                        '; '.join(patient.get('medications', [])),
                        metadata.get('template_path', '').split('/')[-1] if '/' in metadata.get('template_path', '') else metadata.get('template_path', ''),
                        metadata.get('template_path', ''),
                        validation_score,
                        metadata.get('generation_timestamp', '')
                    ])
            
            print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()