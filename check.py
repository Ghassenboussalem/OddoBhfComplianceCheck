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


class TeeOutput:
    """Captures output to both console and file"""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8', errors='replace')
        self.original = original_stream
        
    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.file.flush()
        
    def flush(self):
        self.original.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

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
from datetime import datetime
import uuid

# Load environment
load_dotenv()

if not os.path.exists('.env'):
    print("⚠️  Warning: .env file not found. Some features may be limited.")
    # Don't exit - allow basic checks to work

# Load compliance metrics
try:
    from compliance_metrics import get_compliance_metrics
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Compliance metrics not available: {e}")
    METRICS_AVAILABLE = False

# Load the agent
print("Loading agent...")

# Import HITL components
try:
    from review_manager import ReviewManager, ReviewItem, ReviewStatus
    from audit_logger import AuditLogger
    HITL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  HITL components not available: {e}")
    HITL_AVAILABLE = False

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
    
    Features:
    - AI-enhanced semantic understanding (hybrid mode)
    - AI context-aware mode for false positive elimination (--context-aware=on)
    - Whitelist-based filtering for fund names and strategy terms
    - Data-aware performance checking (only flags actual numbers)
    - Human-in-the-loop review for low-confidence violations
    - Backward compatible with existing workflows
    
    Args:
        json_file_path: Path to JSON document
    
    Returns:
        dict with violations list and JSON output
    """
    try:
        # Initialize metrics tracking
        metrics = None
        doc_start_time = None
        if METRICS_AVAILABLE:
            metrics = get_compliance_metrics()
            doc_start_time = metrics.start_document_check()
        
        # Load document
        with open(json_file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        violations = []
        
        # Track AI usage for metrics
        total_ai_calls = 0
        total_cache_hits = 0
        total_fallbacks = 0

        # Extract metadata
        doc_metadata = doc.get('document_metadata', {})
        fund_isin = doc_metadata.get('fund_isin')
        client_type = doc_metadata.get('client_type', 'retail')
        doc_type = doc_metadata.get('document_type', 'fund_presentation')
        fund_status = doc_metadata.get('fund_status', 'active')
        esg_classification = doc_metadata.get('fund_esg_classification', 'other')
        country_code = doc_metadata.get('country_code', None)
        fund_age_years = doc_metadata.get('fund_age_years', None)
        
        # Check if AI context-aware mode is enabled
        use_ai_context_aware = False
        if hybrid_integration:
            from config_manager import get_config_manager
            config_mgr = get_config_manager()
            config = config_mgr.get_config()
            use_ai_context_aware = config.features.enable_context_aware_ai
            
            if use_ai_context_aware:
                print("🤖 AI Context-Aware Mode: ENABLED")
                print("   Using semantic understanding for false positive elimination\n")

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
        # CHECK 3.1: RISK PROFILE CONSISTENCY (Cross-slide validation)
        # ====================================================================
        print("Checking risk profile consistency...")
        try:
            from check_functions_ai import check_risk_profile_consistency
            risk_consistency_violations = check_risk_profile_consistency(doc)
            violations.extend(risk_consistency_violations)
            if not risk_consistency_violations:
                print("✅ Risk profile consistency: OK\n")
            else:
                print(f"❌ Risk profile consistency: {len(risk_consistency_violations)} violation(s) found\n")
        except Exception as e:
            print(f"⚠️  Risk profile consistency check error: {e}\n")

        # ====================================================================
        # CHECK 3.2: ANGLICISMS IN RETAIL DOCUMENTS
        # ====================================================================
        print("Checking anglicisms in retail documents...")
        try:
            from check_functions_ai import check_anglicisms_retail
            anglicism_violations = check_anglicisms_retail(doc, client_type)
            violations.extend(anglicism_violations)
            if not anglicism_violations:
                print("✅ Anglicisms check: OK\n")
            else:
                print(f"⚠️  Anglicisms: {len(anglicism_violations)} violation(s) found\n")
        except Exception as e:
            print(f"⚠️  Anglicisms check error: {e}\n")

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
                # Use AI context-aware version if enabled
                if use_ai_context_aware:
                    # Import AI-enhanced functions
                    try:
                        from agent import check_repeated_securities_ai, check_prohibited_phrases_ai
                        
                        # Check for repeated securities with whitelist filtering
                        repeated_sec_violations = check_repeated_securities_ai(doc)
                        violations.extend(repeated_sec_violations)
                        
                        # Check for prohibited phrases with context awareness
                        if values_rules:
                            for rule in values_rules:
                                if 'prohibited_phrases' in rule.get('rule_id', '').lower():
                                    phrase_violations = check_prohibited_phrases_ai(doc, rule)
                                    if phrase_violations:
                                        violations.extend(phrase_violations if isinstance(phrase_violations, list) else [phrase_violations])
                        
                        if not repeated_sec_violations:
                            print("✅ Securities/Values (AI Context-Aware): OK\n")
                        else:
                            print(f"⚠️  Securities/Values (AI Context-Aware): {len(repeated_sec_violations)} warning(s)\n")
                    
                    except ImportError as e:
                        print(f"⚠️  AI context-aware functions not available: {e}")
                        print("   Falling back to standard checks...\n")
                        values_violations = check_values_rules_enhanced(doc)
                        violations.extend(values_violations)
                        if not values_violations:
                            print("✅ Securities/Values: OK\n")
                        else:
                            print(f"⚠️  Securities/Values: {len(values_violations)} warning(s)\n")
                    except Exception as e:
                        print(f"⚠️  AI context-aware check error: {e}")
                        print("   Falling back to standard checks...\n")
                        values_violations = check_values_rules_enhanced(doc)
                        violations.extend(values_violations)
                        if not values_violations:
                            print("✅ Securities/Values: OK\n")
                        else:
                            print(f"⚠️  Securities/Values: {len(values_violations)} warning(s)\n")
                else:
                    # Use standard rule-based checks
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
                # Use AI context-aware version if enabled
                if use_ai_context_aware:
                    try:
                        from check_functions_ai import check_performance_disclaimers_ai, check_document_starts_with_performance_ai
                        
                        # Check performance disclaimers with data-aware logic
                        perf_disclaimer_violations = check_performance_disclaimers_ai(doc)
                        violations.extend(perf_disclaimer_violations)
                        
                        # Check if document starts with performance (cover page only)
                        cover_perf_violations = check_document_starts_with_performance_ai(doc)
                        if cover_perf_violations:
                            violations.extend(cover_perf_violations if isinstance(cover_perf_violations, list) else [cover_perf_violations])
                        
                        # Run other performance checks from standard rules
                        perf_violations = check_performance_rules_enhanced(doc, client_type, fund_age_years)
                        # Filter out checks that were already done with AI
                        filtered_perf_violations = [
                            v for v in perf_violations 
                            if 'disclaimer' not in v.get('rule', '').lower() 
                            and 'starts with' not in v.get('message', '').lower()
                        ]
                        violations.extend(filtered_perf_violations)
                        
                        total_perf_violations = len(perf_disclaimer_violations) + len(cover_perf_violations if isinstance(cover_perf_violations, list) else [cover_perf_violations] if cover_perf_violations else []) + len(filtered_perf_violations)
                        
                        if total_perf_violations == 0:
                            print("✅ Performance (AI Context-Aware): OK\n")
                        else:
                            print(f"❌ Performance (AI Context-Aware): {total_perf_violations} violation(s) found\n")
                    
                    except ImportError as e:
                        print(f"⚠️  AI context-aware functions not available: {e}")
                        print("   Falling back to standard checks...\n")
                        perf_violations = check_performance_rules_enhanced(doc, client_type, fund_age_years)
                        violations.extend(perf_violations)
                        if not perf_violations:
                            print("✅ Performance: OK\n")
                        else:
                            print(f"❌ Performance: {len(perf_violations)} violation(s) found\n")
                    except Exception as e:
                        print(f"⚠️  AI context-aware check error: {e}")
                        print("   Falling back to standard checks...\n")
                        perf_violations = check_performance_rules_enhanced(doc, client_type, fund_age_years)
                        violations.extend(perf_violations)
                        if not perf_violations:
                            print("✅ Performance: OK\n")
                        else:
                            print(f"❌ Performance: {len(perf_violations)} violation(s) found\n")
                else:
                    # Use standard rule-based checks
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
            
            # Display AI reasoning if available (from context-aware mode)
            if 'ai_reasoning' in v and v['ai_reasoning']:
                print(f"\n🤖 AI Reasoning:")
                print(f"   {v['ai_reasoning']}")
            
            # Display method used (AI, Rules, or Hybrid)
            if 'method' in v and v['method']:
                print(f"  Method: {v['method']}")
            
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
        # HITL: QUEUE LOW-CONFIDENCE VIOLATIONS FOR REVIEW
        # ====================================================================
        queued_count = 0
        if HITL_AVAILABLE and violations:
            try:
                # Get HITL configuration
                hitl_enabled = False
                review_threshold = 70
                auto_queue = True
                
                if hybrid_integration:
                    from config_manager import get_config_manager
                    config_mgr = get_config_manager()
                    hitl_config = config_mgr.get_hitl_config()
                    hitl_enabled = hitl_config.enabled
                    review_threshold = hitl_config.review_threshold
                    auto_queue = hitl_config.auto_queue_low_confidence
                
                if hitl_enabled and auto_queue:
                    # Initialize review manager and audit logger
                    review_mgr = ReviewManager()
                    audit_logger = AuditLogger()
                    
                    # Generate check session ID
                    check_id = f"check_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
                    
                    # Filter violations that need review (exclude 0 confidence as those are rule-based)
                    violations_for_review = [
                        v for v in violations 
                        if 0 < v.get('confidence', 100) < review_threshold
                    ]
                    
                    if violations_for_review:
                        print(f"\n{'='*70}")
                        print(f"👤 QUEUEING VIOLATIONS FOR HUMAN REVIEW")
                        print(f"{'='*70}")
                        print(f"\nFound {len(violations_for_review)} violation(s) below confidence threshold ({review_threshold}%)")
                        print(f"Adding to review queue...\n")
                        
                        # Queue each violation
                        for idx, v in enumerate(violations_for_review, 1):
                            # Create ReviewItem
                            review_id = f"{check_id}_item_{idx}"
                            
                            review_item = ReviewItem(
                                review_id=review_id,
                                document_id=json_file_path,
                                check_type=v['type'],
                                slide=v['slide'],
                                location=v['location'],
                                predicted_violation=True,
                                confidence=v.get('confidence', 0),
                                ai_reasoning=v.get('ai_reasoning', ''),
                                evidence=v['evidence'],
                                rule=v['rule'],
                                severity=v['severity'],
                                created_at=datetime.now().isoformat(),
                                priority_score=100 - v.get('confidence', 0),  # Lower confidence = higher priority
                                status=ReviewStatus.PENDING
                            )
                            
                            # Add to queue
                            review_mgr.add_to_queue(review_item)
                            queued_count += 1
                            
                            print(f"  ✓ Queued: {v['type']} - {v['message'][:60]}... (confidence: {v.get('confidence', 0)}%)")
                        
                        # Log check operation to audit
                        audit_logger.log_queue_action(
                            action='check_completed',
                            details={
                                'check_id': check_id,
                                'document_id': json_file_path,
                                'check_type': 'full_compliance',
                                'total_violations': len(violations),
                                'queued_for_review': queued_count,
                                'fund_isin': fund_isin,
                                'client_type': client_type,
                                'ai_enabled': hybrid_integration.config.get('ai_enabled', False) if hybrid_integration else False,
                                'review_threshold': review_threshold
                            }
                        )
                        
                        print(f"\n✓ Successfully queued {queued_count} violation(s) for review")
                        print(f"  Run 'python review.py' to start reviewing")
                        print(f"{'='*70}\n")
                    else:
                        print(f"\n✓ All violations have sufficient confidence (≥{review_threshold}%)")
                        print(f"  No items queued for human review\n")
                        
            except Exception as e:
                print(f"\n⚠️  Error queueing violations for review: {e}")
                print(f"  Violations saved to JSON but not queued for interactive review\n")

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

        # Record document metrics
        if METRICS_AVAILABLE and metrics and doc_start_time:
            # Count false positives/negatives if available in violations
            false_positives = sum(1 for v in violations if v.get('is_false_positive', False))
            false_negatives = 0  # Would need ground truth to calculate
            
            # Get AI usage stats from hybrid integration if available
            if hybrid_integration:
                cache_stats = hybrid_integration.get_cache_stats()
                if cache_stats:
                    total_cache_hits = cache_stats.get('hits', 0)
                    total_ai_calls = cache_stats.get('total_requests', 0) - total_cache_hits
            
            metrics.record_document_result(
                document_id=json_file_path,
                start_time=doc_start_time,
                violations=violations,
                false_positives=false_positives,
                false_negatives=false_negatives,
                ai_calls=total_ai_calls,
                cache_hits=total_cache_hits,
                fallback_count=total_fallbacks
            )

        return {
            'total_violations': len(violations),
            'violations': violations,
            'fund_isin': fund_isin,
            'client_type': client_type,
            'json_output': output,
            'queued_for_review': queued_count
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
        print("  --context-aware=on|off  Enable/disable AI context-aware mode (false positive elimination)")
        print("  --ai-confidence=N       Set AI confidence threshold (default: 70)")
        print("  --review-mode           Enter interactive review mode after checking")
        print("  --review-threshold=N    Set review threshold for low-confidence items (default: 70)")
        print("  --show-metrics          Display performance metrics after check")
        print("\nExamples:")
        print("  python check.py exemple.json")
        print("  python check.py exemple.json --hybrid-mode=on")
        print("  python check.py exemple.json --rules-only")
        print("  python check.py exemple.json --context-aware=on")
        print("  python check.py exemple.json --ai-confidence=80 --show-metrics")
        print("  python check.py exemple.json --review-mode")
        print("  python check.py exemple.json --review-threshold=60 --review-mode")
        print("\nFeatures:")
        print("  - AI-enhanced semantic understanding (hybrid mode)")
        print("  - AI context-aware mode for false positive elimination")
        print("  - Whitelist-based filtering for fund names and strategy terms")
        print("  - Data-aware performance checking (only flags actual numbers)")
        print("  - Human-in-the-loop review for low-confidence violations")
        print("  - Interactive review mode with CLI interface")
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
    
    # Set up terminal output logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    terminal_log_file = f"terminal_output_{timestamp}.txt"
    
    # Redirect stdout and stderr to both console and file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = TeeOutput(terminal_log_file, original_stdout)
    sys.stderr = TeeOutput(terminal_log_file, original_stderr)
    
    print(f"{'='*70}")
    print(f"Terminal output will be saved to: {terminal_log_file}")
    print(f"{'='*70}\n")
    
    # Parse command-line options
    show_metrics = False
    review_mode = False
    review_threshold = None
    
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
        elif arg.startswith('--context-aware='):
            mode = arg.split('=')[1].lower()
            if hybrid_integration:
                from config_manager import get_config_manager
                config_mgr = get_config_manager()
                if mode == 'on':
                    # Enable context-aware AI mode
                    config_mgr.update_config(enable_context_aware_ai=True)
                    print("✓ AI Context-Aware mode enabled via command line")
                    print("  This mode uses semantic understanding to eliminate false positives")
                elif mode == 'off':
                    # Disable context-aware AI mode
                    config_mgr.update_config(enable_context_aware_ai=False)
                    print("✓ AI Context-Aware mode disabled via command line")
        elif arg.startswith('--ai-confidence='):
            threshold = int(arg.split('=')[1])
            if hybrid_integration:
                hybrid_integration.update_config(confidence_threshold=threshold)
                print(f"✓ AI confidence threshold set to {threshold}%")
        elif arg == '--review-mode':
            review_mode = True
            print("✓ Interactive review mode enabled")
        elif arg.startswith('--review-threshold='):
            review_threshold = int(arg.split('=')[1])
            print(f"✓ Review threshold set to {review_threshold}%")
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
        if show_metrics:
            if METRICS_AVAILABLE:
                # Display comprehensive compliance metrics
                compliance_metrics = get_compliance_metrics()
                compliance_metrics.print_dashboard()
                
                # Export metrics to file
                metrics_filename = json_file.replace('.json', '_metrics.json')
                compliance_metrics.export_metrics(metrics_filename)
                print(f"\n📊 Metrics exported to: {metrics_filename}")
            elif hybrid_integration and hybrid_integration.is_hybrid_enabled():
                # Fallback to basic hybrid metrics if compliance metrics not available
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
            else:
                print("\n⚠️  Metrics not available (hybrid mode not enabled)")
        
        # Display review mode option if items were queued
        if HITL_AVAILABLE and result.get('queued_for_review', 0) > 0:
            # Enter review mode if requested
            if review_mode:
                print("\n🔍 Entering interactive review mode...")
                print("   Press Ctrl+C to exit\n")
                
                try:
                    # Try to import and start review CLI
                    from review import ReviewCLI
                    
                    cli = ReviewCLI()
                    cli.start_interactive_session(reviewer_id="cli_user")
                    
                except ImportError:
                    print("⚠️  Review CLI not available. Please run 'python review.py' separately.")
                except KeyboardInterrupt:
                    print("\n\n✓ Review mode cancelled by user")
                except Exception as e:
                    print(f"\n⚠️  Error starting review mode: {e}")
                    print("   Please run 'python review.py' separately.")
        
        # Close terminal log files
        if hasattr(sys.stdout, 'close'):
            print(f"\n{'='*70}")
            print(f"📄 Terminal output saved to: {terminal_log_file}")
            print(f"{'='*70}")
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        if result['total_violations'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Close terminal log files on error
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure terminal log is closed even on unexpected errors
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
        raise
