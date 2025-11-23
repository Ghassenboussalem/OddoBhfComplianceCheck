#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
General Checking Tools for Multi-Agent Compliance System

These tools handle general compliance checks:
- Glossary requirements for technical terms
- Morningstar rating date validation
- Source citations for external data
- Technical term identification

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List, Set
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_all_text_from_doc(doc: dict) -> str:
    """Extract all text from document for analysis"""
    all_text = []

    if 'page_de_garde' in doc:
        all_text.append(json.dumps(doc['page_de_garde']))

    if 'slide_2' in doc:
        all_text.append(json.dumps(doc['slide_2']))

    if 'pages_suivantes' in doc:
        for page in doc['pages_suivantes']:
            all_text.append(json.dumps(page))

    if 'page_de_fin' in doc:
        all_text.append(json.dumps(doc['page_de_fin']))

    return '\n'.join(all_text)


# ============================================================================
# GLOSSARY REQUIREMENT TOOL
# ============================================================================

@tool
def check_glossary_requirement(document: dict, client_type: str) -> Optional[dict]:
    """
    Check if technical terms require a glossary for retail documents.

    Regulatory requirement (AMF): Documents for retail (non-professional)
    investors must include a glossary if they contain technical/specialized
    financial terms.

    Technical terms include:
    - Investment strategies: "momentum", "quantitative", "systematic"
    - Metrics: "volatility", "Sharpe ratio", "alpha", "beta"
    - Instruments: "derivatives", "futures", "swaps"
    - Indices: "S&P 500", "MSCI World"

    Args:
        document: Document dictionary
        client_type: Client type (retail/professional)

    Returns:
        Violation dictionary if glossary is required but missing, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        # Professional clients don't need glossary
        if client_type.lower() != 'retail':
            logger.info("Professional document - glossary not required")
            return None

        all_text = extract_all_text_from_doc(document).lower()

        # Define technical terms to check for
        technical_terms = [
            'momentum', 'quantitative', 'quantitatif', 'volatility', 'volatilité',
            's&p 500', 'sri', 'smart momentum', 'systematic', 'systématique',
            'behavioral finance', 'finance comportementale', 'derivatives',
            'dérivés', 'hedge ratio', 'alpha', 'beta', 'sharpe ratio',
            'tracking error', 'duration', 'durée', 'convexity', 'convexité',
            'var', 'value at risk', 'stress test', 'backtesting',
            'overweight', 'underweight', 'surpondération', 'sous-pondération'
        ]

        # Check which technical terms are present
        terms_found = [term for term in technical_terms if term in all_text]

        if not terms_found:
            # No technical terms found - glossary not required
            logger.info("No technical terms found - glossary not required")
            return None

        # Technical terms found - check for glossary
        glossary_keywords = [
            'glossaire', 'glossary', 'lexique', 'définitions',
            'definitions', 'terminologie', 'terminology'
        ]

        has_glossary = any(kw in all_text for kw in glossary_keywords)

        if not has_glossary:
            # Technical terms without glossary - this is a violation
            logger.warning(f"Technical terms found without glossary: {len(terms_found)} terms")

            return {
                'type': 'GENERAL',
                'severity': 'MAJOR',
                'slide': 'End of document',
                'location': 'Missing glossary',
                'rule': 'GEN_006: Retail docs with technical terms need glossary',
                'message': f'Technical terms without glossary: {len(terms_found)} terms',
                'evidence': f'Found: {", ".join(terms_found[:5])}{"..." if len(terms_found) > 5 else ""}',
                'confidence': 90,
                'method': 'RULE_BASED',
                'rule_hints': f'Technical terms: {len(terms_found)}, Glossary: {has_glossary}'
            }

        # Glossary present - no violation
        logger.info(f"Glossary found for {len(terms_found)} technical terms")
        return None

    except Exception as e:
        logger.error(f"Error checking glossary requirement: {e}")
        return {
            'type': 'GENERAL',
            'severity': 'MAJOR',
            'slide': 'End of document',
            'location': 'Glossary section',
            'rule': 'GEN_006: Retail docs with technical terms need glossary',
            'message': f'Error checking glossary: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# MORNINGSTAR DATE TOOL
# ============================================================================

@tool
def check_morningstar_date(document: dict) -> Optional[dict]:
    """
    Check that Morningstar ratings include calculation date.

    Regulatory requirement (AMF): If a Morningstar rating is displayed,
    it MUST include:
    1. The date of the rating ("as of DD/MM/YYYY" or "au DD/MM/YYYY")
    2. The rating must be current (not outdated)
    3. Clear indication it's a Morningstar rating

    Args:
        document: Document dictionary

    Returns:
        Violation dictionary if Morningstar rating lacks date, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        all_text = extract_all_text_from_doc(document)
        all_text_lower = all_text.lower()

        # Check for Morningstar rating presence
        morningstar_keywords = ['morningstar', 'rating', 'étoiles', 'stars', '★']
        has_morningstar = any(kw in all_text_lower for kw in morningstar_keywords)

        if not has_morningstar:
            # No Morningstar rating found - no violation
            logger.info("No Morningstar rating found in document")
            return None

        # Morningstar rating found - check for date
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'as of',  # "as of [date]"
            r'as at',  # "as at [date]"
            r'au \d{1,2}',  # "au DD/MM/YYYY"
            r'à la date du',  # "à la date du DD/MM/YYYY"
            r'arrêté au',  # "arrêté au DD/MM/YYYY"
            r'december \d{4}',  # "December 2024"
            r'décembre \d{4}',  # "décembre 2024"
            r'january \d{4}',  # Month Year format
            r'janvier \d{4}',
            r'february \d{4}',
            r'février \d{4}',
            r'march \d{4}',
            r'mars \d{4}',
            r'april \d{4}',
            r'avril \d{4}',
            r'may \d{4}',
            r'mai \d{4}',
            r'june \d{4}',
            r'juin \d{4}',
            r'july \d{4}',
            r'juillet \d{4}',
            r'august \d{4}',
            r'août \d{4}',
            r'september \d{4}',
            r'septembre \d{4}',
            r'october \d{4}',
            r'octobre \d{4}',
            r'november \d{4}',
            r'novembre \d{4}'
        ]

        has_date = any(re.search(pattern, all_text_lower) for pattern in date_patterns)

        if not has_date:
            # Morningstar rating without date - this is a violation
            logger.warning("Morningstar rating found without date")

            # Try to find the location of Morningstar mention
            location = 'Unknown'
            slide = 'Unknown'

            # Check each section
            if 'morningstar' in json.dumps(document.get('page_de_garde', {})).lower():
                location = 'Cover page'
                slide = 'Cover Page'
            elif 'morningstar' in json.dumps(document.get('slide_2', {})).lower():
                location = 'Slide 2'
                slide = 'Slide 2'
            elif 'pages_suivantes' in document:
                for i, page in enumerate(document['pages_suivantes'], start=3):
                    if 'morningstar' in json.dumps(page).lower():
                        location = f'Slide {page.get("slide_number", i)}'
                        slide = f'Slide {page.get("slide_number", i)}'
                        break

            return {
                'type': 'GENERAL',
                'severity': 'MAJOR',
                'slide': slide,
                'location': location,
                'rule': 'GEN_021: Morningstar rating must include date',
                'message': 'Morningstar rating without date',
                'evidence': 'Rating displayed without as-of date. Must include calculation date.',
                'confidence': 85,
                'method': 'RULE_BASED',
                'rule_hints': f'Morningstar: {has_morningstar}, Date: {has_date}'
            }

        # Morningstar rating with date - no violation
        logger.info("Morningstar rating found with date")
        return None

    except Exception as e:
        logger.error(f"Error checking Morningstar date: {e}")
        return {
            'type': 'GENERAL',
            'severity': 'MAJOR',
            'slide': 'Unknown',
            'location': 'Morningstar rating',
            'rule': 'GEN_021: Morningstar rating must include date',
            'message': f'Error checking Morningstar date: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# SOURCE CITATIONS TOOL
# ============================================================================

@tool
def check_source_citations(document: dict) -> Optional[dict]:
    """
    Check that external data has proper source and date citations.

    Regulatory requirement: Any numerical data, statistic, graph, performance
    table, market study or quantitative analysis must be accompanied by a
    source citation with date.

    Format: 'Source: ODDO BHF Asset Management, as of 31/12/2024' or
    'Source: Bloomberg, December 2024'

    Note: Fund's OWN data (AUM, NAV, fees from prospectus) and prospectus
    data (SRI, allocation) do NOT need source citations.

    Args:
        document: Document dictionary

    Returns:
        Violation dictionary if external data lacks citations, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        all_text = extract_all_text_from_doc(document)
        all_text_lower = all_text.lower()

        # Check for external data indicators
        external_data_indicators = [
            # Market indices
            's&p 500', 'msci', 'stoxx', 'cac 40', 'dax', 'ftse',
            'dow jones', 'nasdaq', 'russell',
            # Data providers
            'bloomberg', 'morningstar', 'refinitiv', 'factset',
            # Market data
            'market data', 'données de marché', 'market statistics',
            'statistiques de marché', 'market study', 'étude de marché',
            # Performance comparisons
            'benchmark', 'indicateur de référence', 'index',
            # Economic data
            'inflation', 'gdp', 'pib', 'unemployment', 'chômage'
        ]

        # Check if document contains external data
        has_external_data = any(indicator in all_text_lower for indicator in external_data_indicators)

        if not has_external_data:
            # No external data found - no citation required
            logger.info("No external data indicators found")
            return None

        # External data found - check for source citations
        source_keywords = [
            'source:', 'source :', 'sources:',
            'as of', 'as at', 'au ', 'à la date du',
            'arrêté au', 'données au', 'chiffres au',
            'data from', 'donnée', 'bloomberg', 'morningstar',
            'oddo bhf asset management', 'oddo bhf am'
        ]

        has_sources = any(kw in all_text_lower for kw in source_keywords)

        # Also check for date patterns near data
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # DD/MM/YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'december \d{4}', r'décembre \d{4}',
            r'january \d{4}', r'janvier \d{4}',
            r'31/12/\d{4}', r'31-12-\d{4}'
        ]

        has_dates = any(re.search(pattern, all_text_lower) for pattern in date_patterns)

        # Find which external data indicators are present
        found_indicators = [ind for ind in external_data_indicators if ind in all_text_lower]

        if not has_sources or not has_dates:
            # External data without proper citations - this is a violation
            logger.warning(f"External data found without proper source/date citations")

            return {
                'type': 'GENERAL',
                'severity': 'MAJOR',
                'slide': 'Multiple locations',
                'location': 'Data sections',
                'rule': 'GEN_003: External data must include source and date',
                'message': 'External data without proper source/date citations',
                'evidence': f'Missing sources for: {", ".join(found_indicators[:3])}{"..." if len(found_indicators) > 3 else ""}',
                'confidence': 80,
                'method': 'RULE_BASED',
                'rule_hints': f'External data: {len(found_indicators)}, Sources: {has_sources}, Dates: {has_dates}'
            }

        # External data with proper citations - no violation
        logger.info(f"External data found with proper citations")
        return None

    except Exception as e:
        logger.error(f"Error checking source citations: {e}")
        return {
            'type': 'GENERAL',
            'severity': 'MAJOR',
            'slide': 'Multiple locations',
            'location': 'Data sections',
            'rule': 'GEN_003: External data must include source and date',
            'message': f'Error checking source citations: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# TECHNICAL TERMS TOOL
# ============================================================================

@tool
def check_technical_terms(document: dict, client_type: str) -> List[str]:
    """
    Identify technical financial terms used in the document.

    This tool identifies specialized financial terminology that may require
    explanation in a glossary for retail investors.

    Technical terms include:
    - Investment strategies: momentum, quantitative, systematic
    - Metrics: volatility, Sharpe ratio, alpha, beta, tracking error
    - Instruments: derivatives, futures, swaps, options
    - Indices: S&P 500, MSCI World, Stoxx 600
    - Jargon: overweight, underweight, hedge ratio

    Args:
        document: Document dictionary
        client_type: Client type (retail/professional)

    Returns:
        List of technical terms found in the document

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        all_text = extract_all_text_from_doc(document).lower()

        # Define comprehensive list of technical terms
        technical_terms_dict = {
            # Investment strategies
            'momentum': 'momentum',
            'quantitative': 'quantitative',
            'quantitatif': 'quantitative',
            'systematic': 'systematic',
            'systématique': 'systematic',
            'smart beta': 'smart beta',
            'factor investing': 'factor investing',
            'value investing': 'value investing',
            'growth investing': 'growth investing',

            # Risk metrics
            'volatility': 'volatility',
            'volatilité': 'volatility',
            'sharpe ratio': 'Sharpe ratio',
            'alpha': 'alpha',
            'beta': 'beta',
            'tracking error': 'tracking error',
            'var': 'VaR',
            'value at risk': 'Value at Risk',
            'duration': 'duration',
            'durée': 'duration',
            'convexity': 'convexity',
            'convexité': 'convexity',

            # Instruments
            'derivatives': 'derivatives',
            'dérivés': 'derivatives',
            'futures': 'futures',
            'options': 'options',
            'swaps': 'swaps',
            'forwards': 'forwards',
            'cds': 'CDS',
            'credit default swap': 'credit default swap',

            # Indices
            's&p 500': 'S&P 500',
            'msci world': 'MSCI World',
            'msci': 'MSCI',
            'stoxx 600': 'Stoxx 600',
            'stoxx': 'Stoxx',
            'cac 40': 'CAC 40',

            # Portfolio management
            'overweight': 'overweight',
            'underweight': 'underweight',
            'surpondération': 'overweight',
            'sous-pondération': 'underweight',
            'hedge ratio': 'hedge ratio',
            'rebalancing': 'rebalancing',
            'rééquilibrage': 'rebalancing',

            # Performance
            'outperformance': 'outperformance',
            'surperformance': 'outperformance',
            'underperformance': 'underperformance',
            'sous-performance': 'underperformance',
            'benchmark': 'benchmark',
            'indicateur de référence': 'benchmark',

            # Other
            'behavioral finance': 'behavioral finance',
            'finance comportementale': 'behavioral finance',
            'backtesting': 'backtesting',
            'stress test': 'stress test',
            'scenario analysis': 'scenario analysis'
        }

        # Find which technical terms are present
        terms_found = []
        for term_key, term_display in technical_terms_dict.items():
            if term_key in all_text:
                if term_display not in terms_found:  # Avoid duplicates
                    terms_found.append(term_display)

        logger.info(f"Found {len(terms_found)} technical terms in document")
        return terms_found

    except Exception as e:
        logger.error(f"Error identifying technical terms: {e}")
        return []


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all general checking tools for easy import
GENERAL_TOOLS = [
    check_glossary_requirement,
    check_morningstar_date,
    check_source_citations,
    check_technical_terms
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
            'document_type': 'fund_presentation'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Systematic momentum strategy'
        },
        'slide_2': {
            'content': 'The fund uses quantitative analysis and smart momentum indicators. Morningstar rating: ★★★★'
        },
        'pages_suivantes': [
            {
                'slide_number': 3,
                'content': 'Performance vs S&P 500 benchmark. Volatility: 12%. Sharpe ratio: 1.5'
            }
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    logger.info("=" * 70)
    logger.info("GENERAL CHECKING TOOLS TEST")
    logger.info("=" * 70)

    # Test glossary requirement
    logger.info("\n1. Check Glossary Requirement (Retail):")
    try:
        result = check_glossary_requirement.func(document=test_doc, client_type="retail")
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
            logger.info(f"      Evidence: {result['evidence']}")
        else:
            logger.info("   ✓ PASS: Glossary present or not required")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test Morningstar date
    logger.info("\n2. Check Morningstar Date:")
    try:
        result = check_morningstar_date.func(document=test_doc)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
            logger.info(f"      Evidence: {result['evidence']}")
        else:
            logger.info("   ✓ PASS: Morningstar date present or not applicable")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test source citations
    logger.info("\n3. Check Source Citations:")
    try:
        result = check_source_citations.func(document=test_doc)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
            logger.info(f"      Evidence: {result['evidence']}")
        else:
            logger.info("   ✓ PASS: Source citations present")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test technical terms identification
    logger.info("\n4. Identify Technical Terms:")
    try:
        terms = check_technical_terms.func(document=test_doc, client_type="retail")
        if terms:
            logger.info(f"   📋 Found {len(terms)} technical terms:")
            for term in terms[:10]:  # Show first 10
                logger.info(f"      - {term}")
            if len(terms) > 10:
                logger.info(f"      ... and {len(terms) - 10} more")
        else:
            logger.info("   ℹ️  No technical terms found")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("All general checking tools tested!")
    logger.info("=" * 70)
