#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Compliance Checker with Fixed Logic
Usage: python check.py <json_file>
"""

import sys
import os
import io
import json

# Fix encoding for ALL output streams
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
else:
    # Unix/Linux
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set environment encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

from dotenv import load_dotenv

# Load environment
load_dotenv()

if not os.path.exists('.env'):
    print("⚠️  Warning: .env file not found. Some features may be limited.")
    # Don't exit - allow basic checks to work

# Load the agent
print("Loading agent...")

# Try to load hybrid integration first
hybrid_integration = None
try:
    from check_hybrid import get_hybrid_integration
    hybrid_integration = get_hybrid_integration()
    if hybrid_integration.is_hybrid_enabled():
        print("✓ Hybrid AI+Rules mode enabled")
    else:
        print("⚠ Hybrid mode not available, using rules only")
except Exception as e:
    print(f"⚠ Hybrid integration not available: {e}")
    print("  Using traditional rule-based checking")

# Load agent functions
# Prefer agent.py which has all enhanced functions
agent_path = 'agent.py'
if not os.path.exists(agent_path):
    agent_path = 'agent_enhanced_ai.py'
    if not os.path.exists(agent_path):
        print(f"ERROR: No agent file found!")
        sys.exit(1)

print(f"Loading agent from {agent_path}...")
with open(agent_path, encoding='utf-8') as f:
    exec(f.read())

def check_document_compliance(json_file_path):
    """
    Full document compliance checker with enhanced logic
    
    Args:
        json_file_path: Path to JSON document
    
    Returns:
        dict with violations list and JSON output
    """
    try:
        # Load document
        with open(json_file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        violations = []

        # Extract metadata
        doc_metadata = doc.get('document_metadata', {})
        fund_isin = doc_metadata.get('fund_isin')
        client_type = doc_metadata.get('client_type', 'retail')
        doc_type = doc_metadata.get('document_type', 'fund_presentation')
        fund_status = doc_metadata.get('fund_status', 'active')
        esg_classification = doc_metadata.get('fund_esg_classification', 'other')
        country_code = doc_metadata.get('country_code', None)
        fund_age_years = doc_metadata.get('fund_age_years', None)

        print(f"\n{'='*70}")
        print(f"📋 COMPLIANCE REPORT - ENHANCED VERSION")
        print(f"{'='*70}")
        print(f"File: {json_file_path}")
        print(f"Fund ISIN: {fund_isin or 'Not specified'}")
        print(f"Client Type: {client_type.upper()}")
        print(f"Document Type: {doc_type}")
        print(f"Fund Status: {fund_status}")
        print(f"ESG Classification: {esg_classification}")
        print(f"{'='*70}\n")

        # ====================================================================
        # CHECK 1: DISCLAIMERS
        # ====================================================================
        if disclaimers_db:
            doc_type_mapping = {
                'fund_presentation': 'OBAM Presentation',
                'commercial_doc': 'Commercial documentation',
                'fact_sheet': 'OBAM Presentation'
            }

            disclaimer_type = doc_type_mapping.get(doc_type, 'OBAM Presentation')

            if disclaimer_type in disclaimers_db:
                client_key = 'professional' if client_type.lower() == 'professional' else 'retail'
                required_disclaimer = disclaimers_db[disclaimer_type].get(client_key)

                if required_disclaimer and len(required_disclaimer) > 50:
                    all_slides_text = extract_all_text_from_doc(doc)
                    result = check_disclaimer_in_document(all_slides_text, required_disclaimer)

                    if result.get('status') == 'MISSING':
                        violations.append({
                            'type': 'DISCLAIMER',
                            'severity': 'CRITICAL',
                            'slide': 'Document-wide',
                            'location': 'Missing',
                            'rule': f'Required {client_key} disclaimer',
                            'message': f"Required disclaimer not found",
                            'evidence': result.get('explanation', 'Standard disclaimer phrases missing'),
                            'confidence': result.get('confidence', 80)
                        })
                    else:
                        print("✅ Disclaimers: OK\n")

        # ====================================================================
        # CHECK 2: REGISTRATION
        # ====================================================================
        if fund_isin and funds_db:
            print("Checking registration...")
            fund_info = None
            for fund in funds_db:
                if fund['isin'] == fund_isin:
                    fund_info = fund
                    break

            if fund_info:
                authorized_countries = fund_info['authorized_countries']
                print(f"  Fund: {fund_info['fund_name']}")
                print(f"  Authorized in {len(authorized_countries)} countries\n")

                reg_violations = check_registration_rules_enhanced(
                    doc,
                    fund_isin,
                    authorized_countries
                )
                violations.extend(reg_violations)

                if not reg_violations:
                    print("✅ Registration compliance: OK\n")
            else:
                print(f"⚠️  Fund {fund_isin} not found in registration database\n")

        # ====================================================================
        # CHECK 3: STRUCTURE
        # ====================================================================
        if structure_rules:
            print("Checking structure...")
            try:
                structure_violations = check_structure_rules_enhanced(doc, client_type, fund_status)
                violations.extend(structure_violations)
                if not structure_violations:
                    print("✅ Structure: OK\n")
                else:
                    print(f"❌ Structure: {len(structure_violations)} violation(s) found\n")
            except Exception as e:
                print(f"⚠️  Structure check error: {e}\n")

        # ====================================================================
        # CHECK 4: GENERAL RULES
        # ====================================================================
        if general_rules:
            print("Checking general rules...")
            try:
                gen_violations = check_general_rules_enhanced(doc, client_type, country_code)
                violations.extend(gen_violations)
                if not gen_violations:
                    print("✅ General rules: OK\n")
                else:
                    print(f"❌ General rules: {len(gen_violations)} violation(s) found\n")
            except Exception as e:
                print(f"⚠️  General rules check error: {e}\n")

        # ====================================================================
        # CHECK 5: VALUES/SECURITIES
        # ====================================================================
        if values_rules:
            print("Checking securities/values...")
            try:
                values_violations = check_values_rules_enhanced(doc)
                violations.extend(values_violations)
                if not values_violations:
                    print("✅ Securities/Values: OK\n")
                else:
                    print(f"⚠️  Securities/Values: {len(values_violations)} warning(s)\n")
            except Exception as e:
                print(f"⚠️  Values check error: {e}\n")

        # ====================================================================
        # CHECK 6: ESG
        # ====================================================================
        if esg_rules:
            print("Checking ESG rules...")
            try:
                esg_violations = check_esg_rules_enhanced(doc, esg_classification, client_type)
                violations.extend(esg_violations)
                if not esg_violations:
                    print("✅ ESG: OK\n")
                else:
                    print(f"❌ ESG: {len(esg_violations)} violation(s) found\n")
            except Exception as e:
                print(f"⚠️  ESG check error: {e}\n")

        # ====================================================================
        # CHECK 7: PERFORMANCE
        # ====================================================================
        if performance_rules:
            print("Checking performance rules...")
            try:
                perf_violations = check_performance_rules_enhanced(doc, client_type, fund_age_years)
                violations.extend(perf_violations)
                if not perf_violations:
                    print("✅ Performance: OK\n")
                else:
                    print(f"❌ Performance: {len(perf_violations)} violation(s) found\n")
            except Exception as e:
                print(f"⚠️  Performance check error: {e}\n")

        # ====================================================================
        # CHECK 8: PROSPECTUS
        # ====================================================================
        if prospectus_data and prospectus_rules:
            print("Checking prospectus compliance...")
            try:
                prosp_violations = check_prospectus_compliance(doc, prospectus_data)
                violations.extend(prosp_violations)
                if not prosp_violations:
                    print("✅ Prospectus: OK\n")
                else:
                    print(f"⚠️  Prospectus: {len(prosp_violations)} item(s) need verification\n")
            except Exception as e:
                print(f"⚠️  Prospectus check error: {e}\n")

        # ====================================================================
        # FINAL REPORT
        # ====================================================================
        print(f"\n{'='*70}")
        if len(violations) == 0:
            print("✅ NO VIOLATIONS FOUND - Document is compliant!")
        else:
            print(f"❌ {len(violations)} VIOLATION(S) FOUND")
        print(f"{'='*70}\n")

        # Display violations
        for i, v in enumerate(violations, 1):
            print(f"{'='*70}")
            print(f"[{v['severity']}] {v['type']} Violation #{i}")
            print(f"{'='*70}")
            print(f"› Rule: {v['rule']}")
            print(f"¸  Issue: {v['message']}")
            print(f"  Location: {v['slide']} - {v['location']}")
            print(f"\n„ Evidence:")
            print(f"   {v['evidence']}")
            print(f"  Confidence: {v.get('confidence', 'N/A')}%")
            print()

        # Summary
        if violations:
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")

            # By type
            type_counts = {}
            for v in violations:
                vtype = v['type']
                type_counts[vtype] = type_counts.get(vtype, 0) + 1

            print(f"\nViolations by type:")
            for vtype, count in sorted(type_counts.items()):
                print(f"   {vtype}: {count}")

            # By severity
            severity_counts = {}
            for v in violations:
                sev = v['severity']
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            print(f"\nViolations by severity:")
            for sev in ['CRITICAL', 'MAJOR', 'WARNING']:
                if sev in severity_counts:
                    print(f"   {sev}: {severity_counts[sev]}")

        # ====================================================================
        # GENERATE JSON OUTPUT
        # ====================================================================
        
        # Use output formatter for consistent JSON structure
        try:
            from output_formatter import create_formatter
            
            # Get formatter configuration from hybrid integration
            formatter_config = {}
            if hybrid_integration:
                formatter_config = {
                    'use_legacy_format': hybrid_integration.config.get('use_legacy_format', False),
                    'ai_enabled': hybrid_integration.config.get('ai_enabled', False)
                }
            
            formatter = create_formatter(formatter_config)
            
            # Format complete output
            output = formatter.format_complete_output(
                json_file_path=json_file_path,
                metadata=doc_metadata,
                violations=violations
            )
            
            # Validate output structure
            if not formatter.validate_output_structure(output):
                print("⚠️  Warning: Output structure validation failed")
            
            # Save JSON output
            output_filename = json_file_path.replace('.json', '_violations.json')
            formatter.save_output(output, output_filename)
            
            print(f"\n{'='*70}")
            print(f"📄 JSON output saved to: {output_filename}")
            if not formatter.legacy_mode and formatter.include_ai_fields:
                print(f"   Format: Enhanced (with AI fields)")
            else:
                print(f"   Format: Legacy (backward compatible)")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"⚠️  Error formatting output: {e}")
            print("   Falling back to basic JSON output")
            
            # Fallback to basic output
            violations_by_category = {}
            for v in violations:
                category = v['type']
                if category not in violations_by_category:
                    violations_by_category[category] = {
                        'count': 0,
                        'violations': []
                    }
                
                violations_by_category[category]['count'] += 1
                violations_by_category[category]['violations'].append({
                    'rule_id': v['rule'].split(':')[0] if ':' in v['rule'] else v['rule'][:20],
                    'rule_text': v['rule'],
                    'severity': v['severity'],
                    'location': v['location'],
                    'slide_number': v['slide'],
                    'evidence': v['evidence'],
                    'message': v['message'],
                    'confidence': v.get('confidence', 0)
                })
            
            output = {
                'document_info': {
                    'filename': json_file_path,
                    'fund_name': doc_metadata.get('fund_name', 'Unknown'),
                    'fund_isin': fund_isin,
                    'client_type': client_type,
                    'document_type': doc_type,
                    'analysis_date': '2025-01-18'
                },
                'summary': {
                    'total_violations': len(violations),
                    'critical_violations': sum(1 for v in violations if v['severity'] == 'CRITICAL'),
                    'major_violations': sum(1 for v in violations if v['severity'] == 'MAJOR'),
                    'warnings': sum(1 for v in violations if v['severity'] == 'WARNING')
                },
                'violations_by_category': violations_by_category,
                'all_violations': violations
            }
            
            output_filename = json_file_path.replace('.json', '_violations.json')
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*70}")
            print(f"📄 JSON output saved to: {output_filename}")
            print(f"{'='*70}\n")

        return {
            'total_violations': len(violations),
            'violations': violations,
            'fund_isin': fund_isin,
            'client_type': client_type,
            'json_output': output
        }

    except FileNotFoundError:
        print(f"\n❌ File '{json_file_path}' not found")
        return {'error': 'File not found'}
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON file: {e}")
        return {'error': 'Invalid JSON'}
    except Exception as e:
        print(f"\n❌ Error checking document: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("FUND COMPLIANCE CHECKER - ENHANCED VERSION")
        print("="*70)
        print("\nUsage:")
        print("  python check.py <json_file> [options]")
        print("\nOptions:")
        print("  --hybrid-mode=on|off    Enable/disable AI+Rules hybrid mode")
        print("  --rules-only            Use only rule-based checking (no AI)")
        print("  --ai-confidence=N       Set AI confidence threshold (default: 70)")
        print("  --show-metrics          Display performance metrics after check")
        print("\nExamples:")
        print("  python check.py exemple.json")
        print("  python check.py exemple.json --hybrid-mode=on")
        print("  python check.py exemple.json --rules-only")
        print("  python check.py exemple.json --ai-confidence=80 --show-metrics")
        print("\nFeatures:")
        print("  - AI-enhanced semantic understanding (hybrid mode)")
        print("  - Fixed promotional document detection")
        print("  - Fixed target audience detection")
        print("  - Enhanced performance violation detection")
        print("  - Correct prospectus matching")
        print("  - JSON output generation")
        print("  - Backward compatible with existing workflows")
        print("\nThe JSON file should contain:")
        print("  - document_metadata with fund_isin, client_type, etc.")
        print("  - page_de_garde, slide_2, pages_suivantes, page_de_fin")
        print("="*70)
        sys.exit(1)

    json_file = sys.argv[1]
    
    # Parse command-line options
    show_metrics = False
    for arg in sys.argv[2:]:
        if arg.startswith('--hybrid-mode='):
            mode = arg.split('=')[1].lower()
            if mode == 'on' and hybrid_integration:
                hybrid_integration.update_config(ai_enabled=True)
                print("✓ Hybrid mode enabled via command line")
            elif mode == 'off' and hybrid_integration:
                hybrid_integration.update_config(ai_enabled=False)
                print("✓ Hybrid mode disabled via command line")
        elif arg == '--rules-only' and hybrid_integration:
            hybrid_integration.update_config(ai_enabled=False)
            print("✓ Rules-only mode enabled via command line")
        elif arg.startswith('--ai-confidence='):
            threshold = int(arg.split('=')[1])
            if hybrid_integration:
                hybrid_integration.update_config(confidence_threshold=threshold)
                print(f"✓ AI confidence threshold set to {threshold}%")
        elif arg == '--show-metrics':
            show_metrics = True
    
    if not os.path.exists(json_file):
        print(f"\n❌ File '{json_file}' not found")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📄 Available JSON files:")
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            for f in json_files:
                print(f"  - {f}")
        else:
            print("  (none found)")
        sys.exit(1)

    result = check_document_compliance(json_file)
    
    if 'error' not in result:
        print(f"\n{'='*70}")
        print("✅ CHECK COMPLETE")
        print(f"{'='*70}")
        
        if result['total_violations'] == 0:
            print("\n🎉 Document is fully compliant!")
        else:
            print(f"\n⚠️  Please review and fix {result['total_violations']} violation(s)")
            print(f"📄 Detailed JSON report: {sys.argv[1].replace('.json', '_violations.json')}")
        
        # Display performance metrics if requested
        if show_metrics and hybrid_integration and hybrid_integration.is_hybrid_enabled():
            print(f"\n{'='*70}")
            print("📊 PERFORMANCE METRICS")
            print(f"{'='*70}")
            
            cache_stats = hybrid_integration.get_cache_stats()
            if cache_stats:
                print(f"\nCache Statistics:")
                print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
                print(f"  Total Requests: {cache_stats.get('total_requests', 0)}")
                print(f"  Cache Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
            
            perf_metrics = hybrid_integration.get_performance_metrics()
            if perf_metrics:
                print(f"\nProcessing Metrics:")
                if 'total_checks' in perf_metrics:
                    print(f"  Total Checks: {perf_metrics['total_checks']}")
                if 'avg_processing_time_ms' in perf_metrics:
                    print(f"  Avg Processing Time: {perf_metrics['avg_processing_time_ms']:.1f}ms")
                if 'ai_calls' in perf_metrics:
                    print(f"  AI API Calls: {perf_metrics['ai_calls']}")
            
            print(f"{'='*70}")
        
        if result['total_violations'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
