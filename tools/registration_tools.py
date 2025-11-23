#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registration Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Registration Checking Tools for Multi-Agent Compliance System

These tools handle registration compliance checks:
- Country authorization validation
- Country extraction from documents
- Fund registration validation
- Country name variation matching

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List, Set, Tuple
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# COUNTRY NAME VARIATIONS
# ============================================================================

# Common country name variations for matching
COUNTRY_VARIATIONS = {
    "United States": ["USA", "US", "United States of America", "États-Unis", "Etats-Unis"],
    "United Kingdom": ["UK", "Great Britain", "GB", "Royaume-Uni"],
    "Germany": ["Deutschland", "Allemagne"],
    "France": ["République française", "Republique francaise"],
    "Spain": ["España", "Espagne"],
    "Italy": ["Italia", "Italie"],
    "Netherlands": ["Holland", "Pays-Bas"],
    "Belgium": ["Belgique", "België"],
    "Switzerland": ["Suisse", "Schweiz", "Svizzera"],
    "Luxembourg": ["Luxemburg"],
    "Austria": ["Österreich", "Autriche"],
    "Portugal": ["Portugalia"],
    "Greece": ["Grèce", "Hellas"],
    "Ireland": ["Irlande", "Éire"],
    "Denmark": ["Danmark", "Danemark"],
    "Sweden": ["Sverige", "Suède"],
    "Norway": ["Norge", "Norvège"],
    "Finland": ["Suomi", "Finlande"],
    "Poland": ["Polska", "Pologne"],
    "Czech Republic": ["Czechia", "République tchèque", "Republique tcheque"],
    "Hungary": ["Magyarország", "Hongrie"],
    "Romania": ["România", "Roumanie"],
    "Bulgaria": ["България", "Bulgarie"],
    "Croatia": ["Hrvatska", "Croatie"],
    "Slovakia": ["Slovensko", "Slovaquie"],
    "Slovenia": ["Slovenija", "Slovénie"],
    "Estonia": ["Eesti", "Estonie"],
    "Latvia": ["Latvija", "Lettonie"],
    "Lithuania": ["Lietuva", "Lituanie"],
    "Malta": ["Malte"],
    "Cyprus": ["Κύπρος", "Chypre"],
}


def normalize_country_name(country: str) -> str:
    """
    Normalize country name by removing common suffixes and cleaning.

    Args:
        country: Country name to normalize

    Returns:
        Normalized country name
    """
    # Remove common suffixes
    country_clean = country.strip()
    country_clean = re.sub(r'\s*\(fund\)\s*$', '', country_clean, flags=re.IGNORECASE)
    country_clean = re.sub(r'\s*\(fonds\)\s*$', '', country_clean, flags=re.IGNORECASE)
    return country_clean.strip()


def match_country_variations(country1: str, country2: str) -> bool:
    """
    Check if two country names match, considering variations.

    Args:
        country1: First country name
        country2: Second country name

    Returns:
        True if countries match (including variations), False otherwise
    """
    # Normalize both names
    c1 = normalize_country_name(country1).lower()
    c2 = normalize_country_name(country2).lower()

    # Direct match
    if c1 == c2:
        return True

    # Substring match (one contains the other)
    if c1 in c2 or c2 in c1:
        return True

    # Check variations
    for canonical, variations in COUNTRY_VARIATIONS.items():
        canonical_lower = canonical.lower()
        variations_lower = [v.lower() for v in variations]

        # Check if both match the same canonical name or its variations
        c1_matches = c1 == canonical_lower or c1 in variations_lower or any(v in c1 for v in variations_lower)
        c2_matches = c2 == canonical_lower or c2 in variations_lower or any(v in c2 for v in variations_lower)

        if c1_matches and c2_matches:
            return True

    return False


# ============================================================================
# EXTRACT COUNTRIES FROM DOCUMENT TOOL
# ============================================================================

@tool
def extract_countries_from_document(document: dict, ai_engine=None) -> Dict[str, List[Dict]]:
    """
    Extract all country mentions from document with context analysis.

    Uses AI to identify country mentions and classify their context:
    - Distribution authorization claims
    - Investment universe references
    - Risk exposure mentions
    - Legal domicile information

    Args:
        document: Document dictionary with all sections
        ai_engine: AI engine for context analysis (optional)

    Returns:
        Dictionary with:
        - countries_found: List of country mentions with context
        - distribution_claims: Countries mentioned for distribution
        - other_mentions: Countries mentioned in other contexts

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        # Extract all text from document
        doc_text = _extract_all_text_from_doc(document)

        # If AI engine available, use it for extraction
        if ai_engine:
            result = _extract_countries_with_ai(doc_text, ai_engine)
            if result:
                return result

        # Fallback: Simple keyword-based extraction
        logger.info("Using rule-based country extraction (AI not available)")
        return _extract_countries_rule_based(doc_text, document)

    except Exception as e:
        logger.error(f"Error extracting countries from document: {e}")
        return {
            'countries_found': [],
            'distribution_claims': [],
            'other_mentions': []
        }


def _extract_all_text_from_doc(document: dict) -> str:
    """Extract all text from document sections"""
    text_parts = []

    # Extract from all sections
    for section_key in ['page_de_garde', 'slide_2', 'page_de_fin']:
        if section_key in document:
            text_parts.append(json.dumps(document[section_key], ensure_ascii=False))

    # Extract from pages_suivantes
    if 'pages_suivantes' in document:
        for slide in document['pages_suivantes']:
            text_parts.append(json.dumps(slide, ensure_ascii=False))

    return ' '.join(text_parts)


def _extract_countries_with_ai(doc_text: str, ai_engine) -> Optional[Dict]:
    """Use AI to extract countries with context analysis"""
    try:
        prompt = f"""Extract ALL countries mentioned in this document and classify their context.

DOCUMENT TEXT (excerpt):
{doc_text[:3000]}

TASK:
1. Extract ALL country names mentioned in the document
2. For each country, determine the context type:
   - "distribution_claim": Mentioned for distribution/authorization ("Authorized in", "Distributed in", "Available in")
   - "investment_universe": Mentioned as investment target ("Invests in", "Exposure to")
   - "risk_exposure": Mentioned in risk context ("Risk related to")
   - "legal_domicile": Mentioned as legal location ("Domiciled in", "Established in")
   - "example": Used as example or illustration
   - "other": Other context

3. Extract the exact phrase where each country is mentioned

Look for distribution phrases like:
- "Autorisé à la distribution en:", "Authorized in:", "Distributed in:"
- "Pays d'autorisation:", "Countries of authorization:"
- "Available to investors in:"
- "Marketed in:", "Commercialisé en:"

Consider variations: "USA" = "United States", "UK" = "United Kingdom", etc.

Respond with JSON:
{{
  "countries_found": [
    {{
      "country_name": "Country name",
      "context_type": "distribution_claim|investment_universe|risk_exposure|legal_domicile|example|other",
      "exact_phrase": "phrase where mentioned",
      "confidence": 0-100
    }}
  ],
  "distribution_claims": ["Country1", "Country2"],
  "other_mentions": ["Country3", "Country4"]
}}

Return ONLY valid JSON:"""

        result = ai_engine.call_llm(prompt)

        if result and isinstance(result, dict):
            return result

        return None

    except Exception as e:
        logger.error(f"Error in AI country extraction: {e}")
        return None


def _extract_countries_rule_based(doc_text: str, document: dict) -> Dict:
    """Rule-based country extraction as fallback"""
    countries_found = []
    distribution_claims = []

    # Common distribution keywords
    distribution_keywords = [
        'autorisé à la distribution en',
        'authorized in',
        'distributed in',
        'available in',
        'marketed in',
        'commercialisé en',
        'pays d\'autorisation',
        'countries of authorization'
    ]

    # Look for distribution claims in back page
    if 'page_de_fin' in document:
        back_page_text = json.dumps(document['page_de_fin'], ensure_ascii=False).lower()

        # Check if any distribution keyword is present
        has_distribution_section = any(kw in back_page_text for kw in distribution_keywords)

        if has_distribution_section:
            # Extract country-like words (capitalized words, common country names)
            # This is a simple heuristic
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', json.dumps(document['page_de_fin'], ensure_ascii=False))

            # Filter to likely country names (length > 3, not common words)
            common_words = {'Document', 'Page', 'Fund', 'Investment', 'Risk', 'Performance'}
            potential_countries = [w for w in words if len(w) > 3 and w not in common_words]

            for country in potential_countries:
                countries_found.append({
                    'country_name': country,
                    'context_type': 'distribution_claim',
                    'exact_phrase': f'Found in authorization section',
                    'confidence': 60
                })
                distribution_claims.append(country)

    return {
        'countries_found': countries_found,
        'distribution_claims': distribution_claims,
        'other_mentions': []
    }


# ============================================================================
# VALIDATE FUND REGISTRATION TOOL
# ============================================================================

@tool
def validate_fund_registration(
    fund_isin: str,
    mentioned_countries: List[str],
    authorized_countries: List[str]
) -> Dict[str, any]:
    """
    Validate that mentioned countries match fund's authorized countries.

    Checks if countries mentioned in the document for distribution are
    actually authorized for the fund based on registration database.

    Args:
        fund_isin: Fund ISIN code
        mentioned_countries: Countries mentioned in document
        authorized_countries: Countries where fund is authorized

    Returns:
        Dictionary with:
        - is_valid: True if all mentioned countries are authorized
        - unauthorized_countries: List of unauthorized countries
        - authorized_count: Number of authorized countries
        - validation_details: Details for each country

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        unauthorized_countries = []
        validation_details = []

        for mentioned_country in mentioned_countries:
            is_authorized = False
            matched_with = None

            # Check against each authorized country
            for auth_country in authorized_countries:
                if match_country_variations(mentioned_country, auth_country):
                    is_authorized = True
                    matched_with = auth_country
                    break

            validation_details.append({
                'country': mentioned_country,
                'is_authorized': is_authorized,
                'matched_with': matched_with
            })

            if not is_authorized:
                unauthorized_countries.append(mentioned_country)

        result = {
            'is_valid': len(unauthorized_countries) == 0,
            'unauthorized_countries': unauthorized_countries,
            'authorized_count': len(authorized_countries),
            'validation_details': validation_details
        }

        logger.info(f"Registration validation for {fund_isin}: {len(unauthorized_countries)} unauthorized countries")
        return result

    except Exception as e:
        logger.error(f"Error validating fund registration: {e}")
        return {
            'is_valid': False,
            'unauthorized_countries': [],
            'authorized_count': 0,
            'validation_details': [],
            'error': str(e)
        }


# ============================================================================
# CHECK COUNTRY AUTHORIZATION TOOL
# ============================================================================

@tool
def check_country_authorization(
    document: dict,
    fund_isin: str,
    authorized_countries: List[str],
    ai_engine=None
) -> Optional[dict]:
    """
    Check if document mentions unauthorized countries for distribution.

    Main registration compliance check that:
    1. Extracts countries mentioned in document
    2. Identifies distribution authorization claims
    3. Validates against authorized countries list
    4. Returns violation if unauthorized countries found

    Args:
        document: Document dictionary
        fund_isin: Fund ISIN code
        authorized_countries: List of countries where fund is authorized
        ai_engine: AI engine for enhanced analysis (optional)

    Returns:
        Violation dictionary if unauthorized countries found, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        if not fund_isin or not authorized_countries:
            logger.warning("Missing fund_isin or authorized_countries")
            return None

        # Extract countries from document
        extraction_result = extract_countries_from_document.invoke({
            'document': document,
            'ai_engine': ai_engine
        })

        distribution_claims = extraction_result.get('distribution_claims', [])

        if not distribution_claims:
            logger.info("No distribution claims found in document")
            return None

        # Validate registration
        validation_result = validate_fund_registration.invoke({
            'fund_isin': fund_isin,
            'mentioned_countries': distribution_claims,
            'authorized_countries': authorized_countries
        })

        unauthorized_countries = validation_result.get('unauthorized_countries', [])

        if not unauthorized_countries:
            logger.info("All mentioned countries are authorized")
            return None

        # Build violation
        country_details = []
        for country in unauthorized_countries:
            # Find details from extraction
            country_info = next(
                (c for c in extraction_result.get('countries_found', [])
                 if c.get('country_name') == country),
                {}
            )

            country_details.append(
                f"   • {country}\n"
                f"     Context: {country_info.get('exact_phrase', 'Distribution claim')}\n"
                f"     Confidence: {country_info.get('confidence', 80)}%"
            )

        violation = {
            'type': 'REGISTRATION',
            'severity': 'CRITICAL',
            'slide': 'Back page',
            'location': 'Authorization list',
            'rule': 'REG_001: Countries must match registration database',
            'message': (
                f"Document mentions {len(unauthorized_countries)} unauthorized "
                f"distribution {'claim' if len(unauthorized_countries) == 1 else 'claims'}:\n"
                + '\n'.join(country_details)
            ),
            'evidence': (
                f"Fund {fund_isin} is only authorized in: "
                f"{', '.join(sorted(authorized_countries)[:10])}"
                f"{'...' if len(authorized_countries) > 10 else ''}"
            ),
            'confidence': 85,
            'method': 'AI_ENHANCED' if ai_engine else 'RULE_BASED',
            'ai_reasoning': f"Found {len(distribution_claims)} distribution claims, {len(unauthorized_countries)} unauthorized"
        }

        logger.warning(f"Registration violation: {len(unauthorized_countries)} unauthorized countries")
        return violation

    except Exception as e:
        logger.error(f"Error checking country authorization: {e}")
        return None


# ============================================================================
# HELPER: FIND COUNTRY IN SLIDES
# ============================================================================

def find_country_in_slides(document: dict, country: str) -> str:
    """
    Find which slide(s) mention a specific country.

    Args:
        document: Document dictionary
        country: Country name to search for

    Returns:
        String describing slide location (e.g., "Slide 3", "Slides 2, 5")
    """
    country_lower = country.lower()
    slides_found = []

    # Check page_de_garde
    if 'page_de_garde' in document:
        text = json.dumps(document['page_de_garde']).lower()
        if country_lower in text:
            slides_found.append(document['page_de_garde'].get('slide_number', 1))

    # Check slide_2
    if 'slide_2' in document:
        text = json.dumps(document['slide_2']).lower()
        if country_lower in text:
            slides_found.append(document['slide_2'].get('slide_number', 2))

    # Check pages_suivantes
    if 'pages_suivantes' in document:
        for slide in document['pages_suivantes']:
            text = json.dumps(slide).lower()
            if country_lower in text:
                slides_found.append(slide.get('slide_number', '?'))

    # Check page_de_fin
    if 'page_de_fin' in document:
        text = json.dumps(document['page_de_fin']).lower()
        if country_lower in text:
            slides_found.append(document['page_de_fin'].get('slide_number', '?'))

    if slides_found:
        # Remove duplicates and sort
        slides_found = sorted(set([s for s in slides_found if s != '?']))
        if len(slides_found) == 1:
            return f"Slide {slides_found[0]}"
        else:
            return f"Slides {', '.join(map(str, slides_found))}"

    return "Multiple slides"


# ============================================================================
# HELPER: EXTRACT MENTION CONTEXT
# ============================================================================

def extract_mention_context(doc_text: str, country: str, context_chars: int = 500) -> List[str]:
    """
    Extract text around country mention for context analysis.

    Args:
        doc_text: Full document text
        country: Country name to find
        context_chars: Number of characters to include before/after mention

    Returns:
        List of context strings (one per mention)
    """
    doc_lower = doc_text.lower()
    country_lower = country.lower()

    contexts = []
    start = 0

    while True:
        pos = doc_lower.find(country_lower, start)
        if pos == -1:
            break

        context_start = max(0, pos - context_chars)
        context_end = min(len(doc_text), pos + len(country) + context_chars)
        context = doc_text[context_start:context_end]
        contexts.append(context)

        start = pos + 1

    return contexts


# ============================================================================
# EXPORT TOOLS
# ============================================================================

# List of all registration checking tools for easy import
REGISTRATION_TOOLS = [
    check_country_authorization,
    extract_countries_from_document,
    validate_fund_registration
]

# Export all tools for agent use
__all__ = [
    'check_country_authorization',
    'extract_countries_from_document',
    'validate_fund_registration',
    'match_country_variations',
    'normalize_country_name',
    'find_country_in_slides',
    'extract_mention_context',
    'COUNTRY_VARIATIONS',
    'REGISTRATION_TOOLS'
]
