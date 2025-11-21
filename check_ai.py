#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Enhanced Compliance Checker Main Script
Usage: python check_ai.py <json_file>
"""

import sys
import os
import io
import json

# Fix encoding
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
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['PYTHONIOENCODING'] = 'utf-8'

from dotenv import load_dotenv
load_dotenv()

# Load the enhanced agent
print("Loading AI-enhanced agent...")
agent_path = 'agent_enhanced_ai.py'
if not os.path.exists(agent_path):
    print(f"ERROR: {agent_path} not found!")
    sys.exit(1)

with open(agent_path, encoding='utf-8') as f:
    exec(f.read())

# Load check functions
print("Loading AI-enhanced check functions...")
check_functions_path = 'check_functions_ai.py'
if not os.path.exists(check_functions_path):
    print(f"ERROR: {check_functions_path} not found!")
    sys.exit(1)

with open(check_functions_path, encoding='utf-8') as f:
    exec(f.read())


def check_document_compliance_ai(json_file_path):
    """
    Full AI-enhanced document compliance checker
    
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
        print(f"📋 AI-ENHANCED COMPLIANCE REPORT")
        print(f"{'='*70}")
        print(f"File: {json_file_path}")
        print(f"Fund ISIN: {fund_isin or 'Not specified'}")
        print(f"Client Type: {client_type.upper()}")
        print(f"Document Type: {doc_type}")
        print(f"AI Mode: {'Token Factory' if tokenfactory_client else 'Gemini' if model else 'Rule-only'}")
        print(f"{'='*70}\n")

        # ====================================================================
        # CHECK 1: STRUCTURE - AI ENHANCED
        # ====================================================================
        print("🤖 Checking structure with AI...")
        
        try:
            # Promotional mention
            print("  → Promotional document mention...", end=" ")
            v = check_promotional_mention_ai(doc)
            if v:
                violations.append(v)
                print(f"❌ ({v['confidence']}% conf, {v['method']})")
            else:
                print("✅")
            
            # Target audience
            print("  → Target audience...", end=" ")
            v = check_target_audience_ai(doc)
            if v:
                violations.append(v)
                print(f"❌ ({v['confidence']}% conf, {v['method']})")
            else:
                print("✅")
            
            # Slide 2 disclaimers
            print("  → Slide 2 disclaimers...", end=" ")
            v = check_disclaimers_slide2_ai(doc)
            if v:
                violations.append(v)
                print(f"❌ ({v['confidence']}% conf, {v['method']})")
            else:
                print("✅")
            
            # Management company
            print("  → Management company mention...", end=" ")
            v = check_management_company_ai(doc)
            if v:
                violations.append(v)
                print(f"❌ ({v['confidence']}% conf, {v['method']})")
            else:
                print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  Structure check error: {e}\n")

        # ====================================================================
        # CHECK 2: PERFORMANCE - AI ENHANCED
        # ====================================================================
        print("🤖 Checking performance with AI...")
        
        try:
            # Performance disclaimers
            print("  → Performance disclaimers...", end=" ")
            perf_violations = check_performance_disclaimers_ai(doc)
            if perf_violations:
                violations.extend(perf_violations)
                print(f"❌ {len(perf_violations)} issue(s)")
                for v in perf_violations:
                    print(f"     • {v['slide']}: {v['message']} ({v['confidence']}% conf)")
            else:
                print("✅")
            
            # Benchmark comparison
            print("  → Benchmark comparison...", end=" ")
            v = check_benchmark_comparison_ai(doc)
            if v:
                violations.append(v)
                print(f"❌ ({v['confidence']}% conf, {v['method']})")
            else:
                print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  Performance check error: {e}\n")

        # ====================================================================
        # CHECK 3: PROSPECTUS - AI ENHANCED
        # ====================================================================
        if prospectus_data:
            print("🤖 Checking prospectus compliance with AI...")
            
            try:
                print("  → Fund name semantic match...", end=" ")
                v = check_prospectus_fund_match_ai(doc, prospectus_data)
                if v:
                    violations.append(v)
                    print(f"❌ ({v['confidence']}% conf, {v['method']})")
                else:
                    print("✅")
                
                print()
            except Exception as e:
                print(f"\n⚠️  Prospectus check error: {e}\n")

        # ====================================================================
        # CHECK 4: GENERAL RULES - AI ENHANCED
        # ====================================================================
        print("🤖 Checking general rules with AI...")
        
        try:
            # Glossary for retail
            if client_type.lower() == 'retail':
                print("  → Glossary requirement...", end=" ")
                v = check_glossary_requirement_ai(doc, client_type)
                if v:
                    violations.append(v)
                    print(f"❌ ({v['confidence']}% conf, {v['method']})")
                else:
                    print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  General rules check error: {e}\n")

        # ====================================================================
        # CHECK 5: REGISTRATION - AI ENHANCED
        # ====================================================================
        if fund_isin and funds_db:
            print("🤖 Checking registration with AI...")
            
            fund_info = None
            for fund in funds_db:
                if fund['isin'] == fund_isin:
                    fund_info = fund
                    break

            if fund_info:
                authorized_countries = fund_info['authorized_countries']
                print(f"  Fund authorized in {len(authorized_countries)} countries")
                print("  → Country authorization list...", end=" ")
                
                reg_violations = check_registration_countries_ai(
                    doc,
                    fund_isin,
                    authorized_countries
                )
                
                if reg_violations:
                    violations.extend(reg_violations)
                    print(f"❌ {len(reg_violations)} issue(s)")
                    for v in reg_violations:
                        print(f"     • {v['message']}")
                else:
                    print("✅")
                
                print()
            else:
                print(f"⚠️  Fund {fund_isin} not in registration database\n")

        # ====================================================================
        # CHECK 6: STRUCTURE SEMANTIC - AI ENHANCED
        # ====================================================================
        print("🤖 Checking structure semantics with AI...")
        
        try:
            print("  → Date validation and consistency...", end=" ")
            struct_violations = check_structure_semantic_ai(doc, client_type)
            if struct_violations:
                violations.extend(struct_violations)
                print(f"❌ {len(struct_violations)} issue(s)")
                for v in struct_violations:
                    print(f"     • {v['message']} ({v['confidence']}% conf)")
            else:
                print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  Structure semantic check error: {e}\n")

        # ====================================================================
        # CHECK 7: GENERAL SEMANTIC - AI ENHANCED
        # ====================================================================
        print("🤖 Checking general rules with AI context...")
        
        try:
            print("  → Morningstar date validation...", end=" ")
            print("  → Technical terms and glossary...", end=" ")
            gen_violations = check_general_semantic_ai(doc, client_type)
            if gen_violations:
                violations.extend(gen_violations)
                print(f"❌ {len(gen_violations)} issue(s)")
                for v in gen_violations:
                    print(f"     • {v['message']} ({v['confidence']}% conf)")
            else:
                print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  General semantic check error: {e}\n")

        # ====================================================================
        # CHECK 8: VALUES/SECURITIES SEMANTIC - AI ENHANCED
        # ====================================================================
        print("🤖 Checking values/securities with AI...")
        
        try:
            print("  → Company mentions and investment advice...", end=" ")
            val_violations = check_values_semantic_ai(doc)
            if val_violations:
                violations.extend(val_violations)
                print(f"❌ {len(val_violations)} issue(s)")
                for v in val_violations:
                    print(f"     • {v['message']} ({v['confidence']}% conf)")
            else:
                print("✅")
            
            print()
        except Exception as e:
            print(f"\n⚠️  Values semantic check error: {e}\n")

        # ====================================================================
        # FINAL REPORT
        # ====================================================================
        print(f"\n{'='*70}")
        if len(violations) == 0:
            print("✅ NO VIOLATIONS FOUND - Document is compliant!")
        else:
            print(f"❌ {len(violations)} VIOLATION(S) FOUND")
        print(f"{'='*70}\n")

        # Display violations with AI insights
        for i, v in enumerate(violations, 1):
            print(f"{'='*70}")
            print(f"[{v['severity']}] {v['type']} Violation #{i}")
            print(f"{'='*70}")
            print(f"› Rule: {v['rule']}")
            print(f"› Issue: {v['message']}")
            print(f"› Location: {v['slide']} - {v['location']}")
            print(f"› Detection: {v.get('method', 'UNKNOWN')}")
            print(f"› Confidence: {v.get('confidence', 'N/A')}%")
            print(f"\n› Evidence:")
            print(f"   {v['evidence']}")
            
            if v.get('ai_reasoning'):
                print(f"\n› AI Analysis:")
                print(f"   {v['ai_reasoning'][:200]}...")
            
            if v.get('rule_hints'):
                print(f"\n› Rule Check:")
                print(f"   {v['rule_hints']}")
            
            print()

        # Summary
        if violations:
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")

            # By detection method
            method_counts = {}
            for v in violations:
                method = v.get('method', 'UNKNOWN')
                method_counts[method] = method_counts.get(method, 0) + 1

            print(f"\nDetection methods:")
            for method, count in sorted(method_counts.items()):
                emoji = "🤖✅" if method == "VERIFIED_BY_BOTH" else "🤖" if "AI" in method else "📏"
                print(f"   {emoji} {method}: {count}")

            # By confidence level
            high_conf = sum(1 for v in violations if v.get('confidence', 0) >= 85)
            med_conf = sum(1 for v in violations if 70 <= v.get('confidence', 0) < 85)
            low_conf = sum(1 for v in violations if v.get('confidence', 0) < 70)
            
            print(f"\nConfidence levels:")
            print(f"   🔴 High (≥85%): {high_conf}")
            print(f"   🟡 Medium (70-84%): {med_conf}")
            print(f"   🟢 Low (<70%): {low_conf}")

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
                'confidence': v.get('confidence', 0),
                'method': v.get('method', 'UNKNOWN'),
                'ai_reasoning': v.get('ai_reasoning', ''),
                'rule_hints': v.get('rule_hints', '')
            })
        
        output = {
            'document_info': {
                'filename': json_file_path,
                'fund_name': doc_metadata.get('fund_name', 'Unknown'),
                'fund_isin': fund_isin,
                'client_type': client_type,
                'document_type': doc_type,
                'analysis_date': '2025-01-18',
                'ai_mode': 'Token Factory' if tokenfactory_client else 'Gemini' if model else 'Rule-only'
            },
            'summary': {
                'total_violations': len(violations),
                'critical_violations': sum(1 for v in violations if v['severity'] == 'CRITICAL'),
                'major_violations': sum(1 for v in violations if v['severity'] == 'MAJOR'),
                'warnings': sum(1 for v in violations if v['severity'] == 'WARNING'),
                'high_confidence_violations': sum(1 for v in violations if v.get('confidence', 0) >= 85),
                'ai_verified_violations': sum(1 for v in violations if v.get('method') == 'VERIFIED_BY_BOTH')
            },
            'violations_by_category': violations_by_category,
            'all_violations': violations
        }
        
        # Save JSON output
        output_filename = json_file_path.replace('.json', '_violations_ai.json')
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
        print("AI-ENHANCED FUND COMPLIANCE CHECKER")
        print("="*70)
        print("\nUsage:")
        print("  python check_ai.py <json_file>")
        print("\nExample:")
        print("  python check_ai.py exemple.json")
        print("\nFeatures:")
        print("  🤖 AI-powered analysis for ALL checks")
        print("  📏 Rule-based validation for accuracy")
        print("  🔍 Semantic understanding (handles variations, typos)")
        print("  💯 Confidence scoring for each violation")
        print("  📊 Detection method tracking (AI vs Rules vs Both)")
        print("  🎯 Context-aware analysis")
        print("  📄 Enhanced JSON output with AI insights")
        print("\nDetection Methods:")
        print("  • VERIFIED_BY_BOTH: AI + Rules agree (highest confidence)")
        print("  • AI_DETECTED: AI found violation rules missed")
        print("  • FALSE_POSITIVE_FILTERED: Rules flagged but AI cleared")
        print("  • RULE_ONLY: AI unavailable, rules only")
        print("\nRequirements:")
        print("  • Token Factory API key OR Gemini API key")
        print("  • Set in .env file")
        print("="*70)
        sys.exit(1)

    json_file = sys.argv[1]
    
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

    result = check_document_compliance_ai(json_file)
    
    if 'error' not in result:
        print(f"\n{'='*70}")
        print("✅ AI-ENHANCED CHECK COMPLETE")
        print(f"{'='*70}")
        
        if result['total_violations'] == 0:
            print("\n🎉 Document is fully compliant!")
            print("   All checks passed with AI verification")
            sys.exit(0)
        else:
            print(f"\n⚠️  Please review and fix {result['total_violations']} violation(s)")
            print(f"📄 Detailed report: {sys.argv[1].replace('.json', '_violations_ai.json')}")
            
            # Show confidence breakdown
            high_conf = sum(1 for v in result['violations'] if v.get('confidence', 0) >= 85)
            if high_conf > 0:
                print(f"\n🔴 {high_conf} high-confidence violations require immediate attention")
            
            sys.exit(1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
