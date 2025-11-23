#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Performance Checking Tools for Multi-Agent Compliance System

These tools handle performance compliance checks:
- Performance disclaimers (data-aware version)
- Document starts with performance check
- Benchmark comparison validation
- Fund age restrictions

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PERFORMANCE DISCLAIMERS TOOL (DATA-AWARE VERSION)
# ============================================================================

@tool
def check_performance_disclaimers(document: dict, config: dict) -> List[dict]:
    """
    Check that ACTUAL performance data has disclaimers (data-aware version).

    This version eliminates false positives by:
    - Only flagging when ACTUAL performance numbers are present (e.g., "15%", "+20%")
    - Ignoring descriptive keywords like "attractive performance", "performance objective"
    - Using semantic matching for disclaimer detection
    - Verifying disclaimer is on SAME slide as performance data

    Regulatory requirement: When performance data is presented, it must be
    accompanied by the disclaimer "Les performances passées ne préjugent pas
    des performances futures" (Past performance is not indicative of future results).

    Args:
        document: Document dictionary with slides
        config: Configuration dictionary with AI settings

    Returns:
        List of violation dictionaries (one per slide with performance data but no disclaimer)

    Requirements: 2.1, 7.2, 7.5
    Rule: PERF_001
    """
    violations = []

    try:
        # Try to use EvidenceExtractor for better accuracy
        try:
            from evidence_extractor import EvidenceExtractor
            from ai_engine import create_ai_engine_from_env

            ai_engine = create_ai_engine_from_env()
            if ai_engine:
                evidence_extractor = EvidenceExtractor(ai_engine)
                use_evidence_extractor = True
            else:
                use_evidence_extractor = False
        except Exception as e:
            logger.warning(f"Could not initialize EvidenceExtractor: {e}. Using fallback.")
            use_evidence_extractor = False

        # Collect slides to check
        slides_to_check = []
        if 'slide_2' in document:
            slides_to_check.append(('Slide 2', document['slide_2']))

        if 'pages_suivantes' in document:
            for i, page in enumerate(document['pages_suivantes'], start=3):
                slide_num = page.get('slide_number', i)
                slides_to_check.append((f'Slide {slide_num}', page))

        # Check each slide
        for slide_name, slide_data in slides_to_check:
            slide_text = json.dumps(slide_data, ensure_ascii=False)

            # Detect actual performance data
            if use_evidence_extractor:
                # Use EvidenceExtractor to find ACTUAL performance data (numbers with %)
                perf_data = evidence_extractor.find_performance_data(slide_text)

                # Only check if ACTUAL performance data is present
                if not perf_data:
                    continue

                # Check for disclaimer on SAME slide
                required_disclaimer = "performances passées ne préjugent pas"
                disclaimer_match = evidence_extractor.find_disclaimer(slide_text, required_disclaimer)

                if not disclaimer_match or not disclaimer_match.found:
                    # Performance data without disclaimer - violation
                    perf_values = [pd.value for pd in perf_data[:3]]
                    perf_contexts = [pd.context[:80] + "..." for pd in perf_data[:3]]

                    violations.append({
                        'type': 'PERFORMANCE',
                        'severity': 'CRITICAL',
                        'slide': slide_name,
                        'location': slide_data.get('title', 'Performance section'),
                        'rule': 'PERF_001: Performance data must have disclaimer',
                        'message': 'Actual performance data without required disclaimer',
                        'evidence': f'Found performance data: {", ".join(perf_values)}. Context: {perf_contexts[0] if perf_contexts else ""}',
                        'confidence': 95,
                        'method': 'AI_EVIDENCE_EXTRACTOR',
                        'ai_reasoning': f'Detected {len(perf_data)} actual performance data points with numerical values. No disclaimer found on same slide.',
                        'rule_hints': f'Performance values: {perf_values}'
                    })
            else:
                # Fallback: Use pattern matching for performance numbers
                slide_text_lower = slide_text.lower()

                # Pattern matching for performance numbers
                perf_patterns = [
                    r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
                    r'[+\-]?\d+\s*%',  # +15%, -3%
                ]

                has_perf_numbers = any(re.search(pattern, slide_text) for pattern in perf_patterns)

                if not has_perf_numbers:
                    continue

                # Check for disclaimer
                disclaimer_keywords = [
                    'performances passées',
                    'past performance',
                    'ne préjugent pas',
                    'not indicative'
                ]

                has_disclaimer = any(kw in slide_text_lower for kw in disclaimer_keywords)

                if not has_disclaimer:
                    violations.append({
                        'type': 'PERFORMANCE',
                        'severity': 'CRITICAL',
                        'slide': slide_name,
                        'location': slide_data.get('title', 'Performance section'),
                        'rule': 'PERF_001: Performance data must have disclaimer',
                        'message': 'Performance data without disclaimer',
                        'evidence': 'Performance numbers found without accompanying disclaimer',
                        'confidence': 85,
                        'method': 'RULE_BASED_FALLBACK',
                        'rule_hints': 'Pattern-based detection found performance numbers'
                    })

        if violations:
            logger.warning(f"Found {len(violations)} performance disclaimer violations")
        else:
            logger.info("All performance data has required disclaimers")

        return violations

    except Exception as e:
        logger.error(f"Error checking performance disclaimers: {e}")
        return [{
            'type': 'PERFORMANCE',
            'severity': 'CRITICAL',
            'slide': 'Multiple slides',
            'location': 'Performance sections',
            'rule': 'PERF_001: Performance data must have disclaimer',
            'message': f'Error checking performance disclaimers: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }]


# ============================================================================
# DOCUMENT STARTS WITH PERFORMANCE TOOL
# ============================================================================

@tool
def check_document_starts_with_performance(document: dict, config: dict) -> List[dict]:
    """
    Check if document starts with ACTUAL performance data on cover page.

    This version eliminates false positives by:
    - Only checking the cover page (page_de_garde)
    - Only flagging when ACTUAL performance numbers are present (e.g., "15%", "+20%")
    - Ignoring descriptive keywords like "attractive performance", "performance objective"
    - Using EvidenceExtractor to detect real performance data

    Regulatory requirement (PERF_001): A commercial document cannot begin with
    performance but must at minimum start with the fund presentation. Performance
    can never represent the central element of a commercial document.

    Args:
        document: Document dictionary with page_de_garde section
        config: Configuration dictionary with AI settings

    Returns:
        List with violation dictionary if document starts with performance, empty list otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: PERF_001
    """
    violations = []

    try:
        # Only check the cover page (page_de_garde)
        if 'page_de_garde' not in document:
            # No cover page found - cannot violate this rule
            logger.info("No cover page found - skipping check")
            return violations

        cover_page = document['page_de_garde']
        cover_text = json.dumps(cover_page, ensure_ascii=False)

        # Try to use EvidenceExtractor for better accuracy
        try:
            from evidence_extractor import EvidenceExtractor
            from ai_engine import create_ai_engine_from_env

            ai_engine = create_ai_engine_from_env()
            if ai_engine:
                evidence_extractor = EvidenceExtractor(ai_engine)
                use_evidence_extractor = True
            else:
                use_evidence_extractor = False
        except Exception as e:
            logger.warning(f"Could not initialize EvidenceExtractor: {e}. Using fallback.")
            use_evidence_extractor = False

        # Detect actual performance data on cover
        if use_evidence_extractor:
            # Use EvidenceExtractor to find ACTUAL performance data (numbers with %)
            perf_data = evidence_extractor.find_performance_data(cover_text)

            # Only flag if ACTUAL performance numbers are on cover page
            if perf_data:
                # Filter out low-confidence detections (likely descriptive text)
                high_confidence_perf = [pd for pd in perf_data if pd.confidence >= 60]

                if high_confidence_perf:
                    # Actual performance data found on cover page - violation
                    perf_values = [pd.value for pd in high_confidence_perf[:3]]
                    perf_contexts = [pd.context[:80] + "..." for pd in high_confidence_perf[:3]]

                    violations.append({
                        'type': 'PERFORMANCE',
                        'severity': 'MAJOR',
                        'slide': 'Cover Page',
                        'location': 'Beginning of document',
                        'rule': 'PERF_001: Document cannot start with performance data',
                        'message': f'Document starts with performance data ({high_confidence_perf[0].value})',
                        'evidence': f'Found performance data on cover: {", ".join(perf_values)}. Context: {perf_contexts[0] if perf_contexts else ""}. Performance must be preceded by fund presentation.',
                        'confidence': high_confidence_perf[0].confidence,
                        'method': 'AI_EVIDENCE_EXTRACTOR',
                        'ai_reasoning': f'Detected {len(high_confidence_perf)} actual performance data points with numerical values on cover page. Documents should not start with performance data.',
                        'rule_hints': f'Performance values on cover: {perf_values}'
                    })
        else:
            # Fallback: Use pattern matching for performance numbers
            cover_text_lower = cover_text.lower()

            # Pattern matching for performance numbers
            perf_patterns = [
                r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
                r'[+\-]?\d+\s*%',  # +15%, -3%
            ]

            has_perf_numbers = any(re.search(pattern, cover_text) for pattern in perf_patterns)

            if has_perf_numbers:
                violations.append({
                    'type': 'PERFORMANCE',
                    'severity': 'MAJOR',
                    'slide': 'Cover Page',
                    'location': 'Beginning of document',
                    'rule': 'PERF_001: Document cannot start with performance data',
                    'message': 'Document starts with performance data',
                    'evidence': 'Performance numbers found on cover page. Performance must be preceded by fund presentation.',
                    'confidence': 85,
                    'method': 'RULE_BASED_FALLBACK',
                    'rule_hints': 'Pattern-based detection found performance numbers on cover'
                })

        if violations:
            logger.warning("Document starts with performance data on cover page")
        else:
            logger.info("Document does not start with performance data")

        return violations

    except Exception as e:
        logger.error(f"Error checking document start: {e}")
        return [{
            'type': 'PERFORMANCE',
            'severity': 'MAJOR',
            'slide': 'Cover Page',
            'location': 'Beginning of document',
            'rule': 'PERF_001: Document cannot start with performance data',
            'message': f'Error checking document start: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }]


# ============================================================================
# BENCHMARK COMPARISON TOOL
# ============================================================================

@tool
def check_benchmark_comparison(document: dict, config: dict) -> Optional[dict]:
    """
    Check that performance is compared to benchmark.

    Regulatory requirement (PERF_014): Performance must be permanently and
    mandatorily compared to the fund's reference indicator if one exists.
    This requires:
    1. Clear identification of the benchmark
    2. Visual comparison (chart/table)
    3. Side-by-side data

    Args:
        document: Document dictionary
        config: Configuration dictionary with AI settings

    Returns:
        Violation dictionary if performance lacks benchmark comparison, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: PERF_014
    """
    try:
        # Extract all text from document
        all_text_parts = []

        if 'page_de_garde' in document:
            all_text_parts.append(json.dumps(document['page_de_garde']))

        if 'slide_2' in document:
            all_text_parts.append(json.dumps(document['slide_2']))

        if 'pages_suivantes' in document:
            for page in document['pages_suivantes']:
                all_text_parts.append(json.dumps(page))

        if 'page_de_fin' in document:
            all_text_parts.append(json.dumps(document['page_de_fin']))

        all_text = '\n'.join(all_text_parts).lower()

        # Check for performance mentions
        performance_keywords = ['performance', 'rendement', 'surperform', 'return']
        has_performance = any(word in all_text for word in performance_keywords)

        if not has_performance:
            # No performance data mentioned - no violation
            logger.info("No performance data found in document")
            return None

        # Check for benchmark mentions
        benchmark_keywords = [
            's&p 500',
            'msci',
            'benchmark',
            'indicateur de référence',
            'reference indicator',
            'index',
            'indice'
        ]
        has_benchmark = any(kw in all_text for kw in benchmark_keywords)

        # Check for comparison visualization
        chart_keywords = [
            'chart',
            'tableau',
            'graphique',
            'graph',
            'comparison',
            'comparaison',
            'vs',
            'versus'
        ]
        has_chart = any(kw in all_text for kw in chart_keywords)

        # Violation if performance is mentioned but no benchmark comparison
        if has_performance and not (has_benchmark and has_chart):
            logger.warning("Performance data without benchmark comparison")

            return {
                'type': 'PERFORMANCE',
                'severity': 'MAJOR',
                'slide': 'Multiple slides',
                'location': 'Performance sections',
                'rule': 'PERF_014: Performance must compare to benchmark',
                'message': 'Performance without benchmark comparison',
                'evidence': f'Performance mentioned: {has_performance}, Benchmark: {has_benchmark}, Chart/comparison: {has_chart}',
                'confidence': 85,
                'method': 'RULE_BASED',
                'rule_hints': 'Performance claims without clear benchmark chart or comparison'
            }

        # Benchmark comparison present - no violation
        logger.info("Performance data includes benchmark comparison")
        return None

    except Exception as e:
        logger.error(f"Error checking benchmark comparison: {e}")
        return {
            'type': 'PERFORMANCE',
            'severity': 'MAJOR',
            'slide': 'Multiple slides',
            'location': 'Performance sections',
            'rule': 'PERF_014: Performance must compare to benchmark',
            'message': f'Error checking benchmark comparison: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# FUND AGE RESTRICTIONS TOOL
# ============================================================================

@tool
def check_fund_age_restrictions(document: dict, metadata: dict) -> Optional[dict]:
    """
    Check fund age restrictions for performance display.

    Regulatory requirements:
    - PERF_011: Funds with less than one year of performance history cannot
      display performance under any circumstances
    - PERF_007: If the fund has less than 3 years of history, cumulative
      performance is not displayed (except YTD and MTD)

    Args:
        document: Document dictionary
        metadata: Extracted metadata with fund age information

    Returns:
        Violation dictionary if fund age restrictions are violated, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rules: PERF_011, PERF_007
    """
    try:
        # Get fund age from metadata
        fund_age_years = metadata.get('fund_age_years')
        fund_inception_date = metadata.get('fund_inception_date')

        # If no age information, try to calculate from inception date
        if fund_age_years is None and fund_inception_date:
            try:
                from datetime import datetime
                inception = datetime.strptime(str(fund_inception_date), '%Y-%m-%d')
                age_days = (datetime.now() - inception).days
                fund_age_years = age_days / 365.25
            except Exception as e:
                logger.warning(f"Could not calculate fund age from inception date: {e}")

        if fund_age_years is None:
            # Cannot determine fund age - skip check
            logger.info("Fund age not available - skipping age restriction check")
            return None

        # Extract all text to check for performance mentions
        all_text_parts = []

        if 'slide_2' in document:
            all_text_parts.append(json.dumps(document['slide_2']))

        if 'pages_suivantes' in document:
            for page in document['pages_suivantes']:
                all_text_parts.append(json.dumps(page))

        all_text = '\n'.join(all_text_parts).lower()

        # Check for performance data
        perf_patterns = [
            r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
            r'[+\-]?\d+\s*%',  # +15%, -3%
        ]
        has_perf_numbers = any(re.search(pattern, all_text) for pattern in perf_patterns)

        # PERF_011: Funds < 1 year cannot display any performance
        if fund_age_years < 1.0 and has_perf_numbers:
            logger.warning(f"Fund age {fund_age_years:.1f} years < 1 year, but performance data found")

            return {
                'type': 'PERFORMANCE',
                'severity': 'CRITICAL',
                'slide': 'Multiple slides',
                'location': 'Performance sections',
                'rule': 'PERF_011: Funds < 1 year cannot display performance',
                'message': f'Fund is {fund_age_years:.1f} years old (< 1 year) but displays performance',
                'evidence': f'Fund age: {fund_age_years:.1f} years. Performance data found in document.',
                'confidence': 95,
                'method': 'RULE_BASED',
                'rule_hints': 'Remove all performance data - fund too young'
            }

        # PERF_007: Funds < 3 years cannot display cumulative performance (except YTD/MTD)
        if fund_age_years < 3.0:
            cumulative_keywords = [
                'cumulative',
                'cumulé',
                'total return',
                'rendement total'
            ]
            has_cumulative = any(kw in all_text for kw in cumulative_keywords)

            # Check if it's YTD or MTD (which are allowed)
            ytd_mtd_keywords = ['ytd', 'mtd', 'year to date', 'month to date']
            is_ytd_mtd = any(kw in all_text for kw in ytd_mtd_keywords)

            if has_cumulative and not is_ytd_mtd:
                logger.warning(f"Fund age {fund_age_years:.1f} years < 3 years, but cumulative performance found")

                return {
                    'type': 'PERFORMANCE',
                    'severity': 'MAJOR',
                    'slide': 'Multiple slides',
                    'location': 'Performance sections',
                    'rule': 'PERF_007: Funds < 3 years cannot display cumulative performance',
                    'message': f'Fund is {fund_age_years:.1f} years old (< 3 years) but displays cumulative performance',
                    'evidence': f'Fund age: {fund_age_years:.1f} years. Cumulative performance found (not YTD/MTD).',
                    'confidence': 90,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Remove cumulative performance or use only YTD/MTD'
                }

        # No age restriction violations
        logger.info(f"Fund age {fund_age_years:.1f} years - no age restriction violations")
        return None

    except Exception as e:
        logger.error(f"Error checking fund age restrictions: {e}")
        return {
            'type': 'PERFORMANCE',
            'severity': 'MAJOR',
            'slide': 'Multiple slides',
            'location': 'Performance sections',
            'rule': 'PERF_011/PERF_007: Fund age restrictions',
            'message': f'Error checking fund age restrictions: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all performance checking tools for easy import
PERFORMANCE_TOOLS = [
    check_performance_disclaimers,
    check_document_starts_with_performance,
    check_benchmark_comparison,
    check_fund_age_restrictions
]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test with example document
    test_doc = {
        'document_metadata': {
            'fund_isin': 'FR0010135103',
            'fund_name': 'ODDO BHF Algo Trend US Fund',
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_inception_date': '2020-01-15'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel'
        },
        'slide_2': {
            'content': 'Performance: +15.5% in 2023. Les performances passées ne préjugent pas des performances futures.'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'title': 'Performance History',
                'content': 'Annual returns: 2023: +15.5%, 2022: +10.2%, 2021: +8.7%. Benchmark (S&P 500): 2023: +12.1%, 2022: +9.5%, 2021: +7.3%. Chart showing comparison.'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    test_metadata = {
        'fund_age_years': 4.5,
        'fund_inception_date': '2020-01-15'
    }

    test_config = {
        'ai_enabled': True
    }

    logger.info("=" * 70)
    logger.info("PERFORMANCE CHECKING TOOLS TEST")
    logger.info("=" * 70)

    # Test performance disclaimers
    logger.info("\n1. Check Performance Disclaimers:")
    try:
        results = check_performance_disclaimers.func(document=test_doc, config=test_config)
        if results:
            logger.info(f"   ❌ {len(results)} VIOLATION(S):")
            for r in results:
                logger.info(f"      - {r['message']}")
        else:
            logger.info("   ✓ PASS: All performance data has disclaimers")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test document starts with performance
    logger.info("\n2. Check Document Starts with Performance:")
    try:
        results = check_document_starts_with_performance.func(document=test_doc, config=test_config)
        if results:
            logger.info(f"   ❌ VIOLATION: {results[0]['message']}")
        else:
            logger.info("   ✓ PASS: Document does not start with performance")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test benchmark comparison
    logger.info("\n3. Check Benchmark Comparison:")
    try:
        result = check_benchmark_comparison.func(document=test_doc, config=test_config)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Performance includes benchmark comparison")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test fund age restrictions
    logger.info("\n4. Check Fund Age Restrictions:")
    try:
        result = check_fund_age_restrictions.func(document=test_doc, metadata=test_metadata)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Fund age restrictions satisfied")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("All performance checking tools tested!")
    logger.info("=" * 70)
