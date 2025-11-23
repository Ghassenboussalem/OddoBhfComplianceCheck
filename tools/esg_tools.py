#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esg Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
ESG Checking Tools for Multi-Agent Compliance System

These tools handle ESG compliance checks:
- ESG classification validation
- Content distribution analysis
- SFDR compliance checking
- ESG terminology validation

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
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

def extract_all_text_from_doc(document: dict) -> str:
    """Extract all text from document for analysis."""
    text_parts = []

    # Extract from all sections
    if 'page_de_garde' in document:
        text_parts.append(json.dumps(document['page_de_garde'], ensure_ascii=False))

    if 'slide_2' in document:
        text_parts.append(json.dumps(document['slide_2'], ensure_ascii=False))

    if 'pages_suivantes' in document:
        for slide in document['pages_suivantes']:
            text_parts.append(json.dumps(slide, ensure_ascii=False))

    if 'page_de_fin' in document:
        text_parts.append(json.dumps(document['page_de_fin'], ensure_ascii=False))

    return ' '.join(text_parts)


def extract_section_text(document: dict, section_name: str) -> str:
    """Extract text from a specific section."""
    section = document.get(section_name, {})
    return json.dumps(section, ensure_ascii=False)


def analyze_esg_content_with_ai(slide_text: str, slide_number: int, ai_engine) -> dict:
    """
    Analyze if slide is primarily about ESG/sustainability using AI.

    Args:
        slide_text: Full slide text
        slide_number: Slide position
        ai_engine: AI engine instance for analysis

    Returns:
        dict with is_esg_slide, esg_percentage, content_type, esg_elements_found, confidence
    """
    if not ai_engine:
        # Fallback to keyword counting
        esg_keywords = ['esg', 'environmental', 'social', 'governance', 'sustainability', 'sustainable']
        keyword_count = sum(1 for kw in esg_keywords if kw in slide_text.lower())
        is_esg = keyword_count >= 3
        return {
            'is_esg_slide': is_esg,
            'esg_percentage': 80 if is_esg else 20,
            'content_type': 'esg_detailed' if is_esg else 'non_esg',
            'esg_elements_found': [],
            'confidence': 60
        }

    prompt = f"""Analyze if this slide is PRIMARILY about ESG/Sustainability content.

SLIDE {slide_number} CONTENT:
{slide_text[:1500]}

ESG/SUSTAINABILITY INDICATORS:
- English: ESG, environmental, social, governance, sustainability, sustainable, responsible investment, SRI, green, climate, carbon, impact, exclusion, engagement, stewardship, SFDR, Article 8, Article 9, taxonomy, SDG
- French: ESG, environnemental, social, gouvernance, développement durable, investissement responsable, ISR, vert, climat, carbone, impact, exclusion, engagement, SFDR, Article 8, Article 9, taxonomie, ODD

Question: What percentage of this slide's content is dedicated to ESG?

Categories:
- 0-20%: Brief mention, passing reference
- 21-50%: Significant mention but not primary focus
- 51-80%: Major focus, substantial ESG content
- 81-100%: Entirely dedicated to ESG

Respond JSON:
{{
  "is_esg_slide": true/false,
  "esg_percentage": 0-100,
  "content_type": "brief_mention" or "baseline_exclusions" or "esg_strategy" or "esg_detailed" or "non_esg",
  "esg_elements_found": ["list of ESG topics"],
  "confidence": 0-100
}}"""

    try:
        response = ai_engine.client.chat.completions.create(
            model=ai_engine.model,
            messages=[
                {"role": "system", "content": "You are an ESG content analyst. Assess what percentage of content is ESG-related. Be precise in classification. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)

        return result
    except Exception as e:
        logger.error(f"AI error analyzing ESG on slide {slide_number}: {e}")
        return {
            'is_esg_slide': False,
            'esg_percentage': 0,
            'content_type': 'non_esg',
            'esg_elements_found': [],
            'confidence': 50
        }


def check_esg_baseline_with_ai(doc_text: str, ai_engine) -> dict:
    """
    Check if ESG content goes beyond baseline exclusions using AI.

    Args:
        doc_text: Document text
        ai_engine: AI engine instance for analysis

    Returns:
        dict with beyond_baseline, esg_topics_found, confidence, reasoning
    """
    if not ai_engine:
        return {'beyond_baseline': True, 'confidence': 50, 'esg_topics_found': [], 'reasoning': 'No AI available'}

    prompt = f"""Analyze if ESG content in this document goes BEYOND baseline exclusions.

DOCUMENT TEXT (excerpt):
{doc_text[:2500]}

BASELINE EXCLUSIONS (Allowed for all funds):
- English: "controversial weapons exclusion", "baseline exclusions", "firm-wide exclusions", "OBAM exclusion policy"
- French: "exclusion armes controversées", "exclusions socle", "exclusions de base", "politique d'exclusion OBAM"

BEYOND BASELINE (Not allowed for "Other" funds):
- Detailed ESG strategy, ESG integration methodology
- ESG scoring, ESG ratings, ESG analysis
- Climate strategy, carbon reduction targets
- Impact measurement, SDG alignment
- ESG engagement, stewardship activities
- SFDR classification details (Article 8/9)

Question: Does ESG content go beyond simple baseline exclusions?

Respond JSON:
{{
  "mentions_esg": true/false,
  "beyond_baseline": true/false,
  "baseline_only": true/false,
  "esg_topics_found": ["list of topics beyond baseline"],
  "confidence": 0-100,
  "reasoning": "explanation"
}}"""

    try:
        response = ai_engine.client.chat.completions.create(
            model=ai_engine.model,
            messages=[
                {"role": "system", "content": "You are an ESG compliance expert. Distinguish between baseline exclusions and substantive ESG content. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=600
        )

        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace('```json', '').replace('```', '').strip()
        result = json.loads(result_text)

        return result
    except Exception as e:
        logger.error(f"AI error checking baseline: {e}")
        return {'beyond_baseline': True, 'confidence': 50, 'esg_topics_found': [], 'reasoning': f'Error: {e}'}


# ============================================================================
# ESG CLASSIFICATION TOOL
# ============================================================================

@tool
def check_esg_classification(
    document: dict,
    esg_classification: str,
    client_type: str,
    ai_engine=None
) -> Optional[dict]:
    """
    Check ESG classification compliance based on fund's ESG approach.

    ESG Approaches:
    - Engaging: ≥20% exclusion and ≥90% portfolio coverage → unlimited ESG communication
    - Reduced: Limited ESG integration → ESG content must be <10% of strategy presentation
    - Prospectus-limited: Minimal ESG → NO ESG mention in retail documents
    - Other: No specific ESG approach → Only OBAM baseline exclusions allowed

    Args:
        document: Document dictionary
        esg_classification: Fund's ESG classification (engaging, reduced, prospectus_limited, other)
        client_type: Client type (retail, professional)
        ai_engine: AI engine instance for content analysis

    Returns:
        Violation dictionary if classification rules are violated, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: ESG_001, ESG_002, ESG_003, ESG_004, ESG_005
    """
    try:
        classification_lower = esg_classification.lower()

        # Professional funds are exempt from ESG rules (ESG_006)
        if client_type.lower() == 'professional':
            logger.info("Professional fund - ESG rules do not apply")
            return None

        # Engaging approach - no restrictions (ESG_002)
        if 'engaging' in classification_lower or 'engageante' in classification_lower:
            logger.info("Engaging approach - unlimited ESG communication allowed")
            return None

        # For other classifications, we need to analyze content
        # This will be handled by check_content_distribution and check_sfdr_compliance
        logger.info(f"ESG classification: {esg_classification} - will check content restrictions")
        return None

    except Exception as e:
        logger.error(f"Error checking ESG classification: {e}")
        return None


# ============================================================================
# CONTENT DISTRIBUTION TOOL
# ============================================================================

@tool
def check_content_distribution(
    document: dict,
    esg_classification: str,
    client_type: str,
    ai_engine=None
) -> Optional[dict]:
    """
    Check ESG content distribution compliance.

    Validates that ESG content volume complies with fund's classification:
    - Reduced approach: ESG content must be <10% of strategy presentation
    - Prospectus-limited: NO ESG content allowed in retail documents
    - Other: Only baseline exclusions allowed

    Args:
        document: Document dictionary
        esg_classification: Fund's ESG classification
        client_type: Client type (retail, professional)
        ai_engine: AI engine instance for content analysis

    Returns:
        Violation dictionary if content distribution violates rules, None otherwise

    Requirements: 2.1, 7.2, 7.5
    Rule: ESG_003, ESG_004, ESG_005
    """
    try:
        classification_lower = esg_classification.lower()

        # Professional funds are exempt
        if client_type.lower() == 'professional':
            return None

        # Engaging approach has no restrictions
        if 'engaging' in classification_lower or 'engageante' in classification_lower:
            return None

        # Analyze document for ESG content
        logger.info("Analyzing ESG content distribution...")
        esg_slides = []
        total_chars = 0
        esg_chars = 0

        # Check all slides
        all_sections = []
        if 'page_de_garde' in document:
            all_sections.append(('page_de_garde', 1))
        if 'slide_2' in document:
            all_sections.append(('slide_2', 2))
        if 'pages_suivantes' in document:
            for idx, slide in enumerate(document['pages_suivantes'], start=3):
                slide_num = slide.get('slide_number', idx)
                all_sections.append((slide, slide_num))
        if 'page_de_fin' in document:
            all_sections.append(('page_de_fin', 99))

        for section, slide_num in all_sections:
            if isinstance(section, str):
                slide_text = extract_section_text(document, section)
            else:
                slide_text = json.dumps(section, ensure_ascii=False)

            slide_chars = len(slide_text)
            total_chars += slide_chars

            # Analyze with AI
            result = analyze_esg_content_with_ai(slide_text, slide_num, ai_engine)

            if result.get('is_esg_slide') or result.get('esg_percentage', 0) > 20:
                esg_slides.append({
                    'slide': slide_num,
                    'esg_percentage': result.get('esg_percentage', 0),
                    'content_type': result.get('content_type'),
                    'elements': result.get('esg_elements_found', []),
                    'chars': slide_chars,
                    'confidence': result.get('confidence')
                })

                # Calculate ESG chars for this slide
                esg_chars += int(slide_chars * result.get('esg_percentage', 0) / 100)

        esg_volume_percentage = (esg_chars / total_chars * 100) if total_chars > 0 else 0

        logger.info(f"ESG Content Analysis: {esg_chars:,}/{total_chars:,} chars ({esg_volume_percentage:.1f}%)")
        logger.info(f"ESG slides found: {len(esg_slides)}")

        # ESG_003: Reduced approach (< 10% volume)
        if 'reduced' in classification_lower or 'réduite' in classification_lower:
            if esg_volume_percentage >= 10:
                return {
                    'type': 'ESG',
                    'severity': 'MAJOR',
                    'slide': f"{len(esg_slides)} slides with ESG content",
                    'location': 'Document-wide',
                    'rule': 'ESG_003: Reduced approach → ESG content limited to <10% of strategy presentation',
                    'message': f"ESG content exceeds 10% limit for Reduced approach: {esg_volume_percentage:.1f}%",
                    'evidence': f"Total: {total_chars:,} chars, ESG: {esg_chars:,} chars. Must be < 10%. Primarily on slides: {', '.join([str(s['slide']) for s in esg_slides[:5]])}",
                    'confidence': 85,
                    'method': 'AI_ENHANCED',
                    'ai_reasoning': f'Analyzed {len(all_sections)} slides, found ESG content on {len(esg_slides)} slides'
                }

        # ESG_004: Prospectus-limited (NO ESG for retail)
        if ('prospectus' in classification_lower or 'limitée' in classification_lower or 'limited' in classification_lower) and client_type.lower() == 'retail':
            if esg_slides:
                esg_elements = []
                for slide in esg_slides[:5]:
                    esg_elements.extend(slide.get('elements', []))

                # Convert to list and get unique elements
                unique_elements = list(set(esg_elements))[:8]

                return {
                    'type': 'ESG',
                    'severity': 'CRITICAL',
                    'slide': f"{len(esg_slides)} slides",
                    'location': ', '.join([f"Slide {s['slide']}" for s in esg_slides[:5]]),
                    'rule': 'ESG_004: Prospectus-limited approach → NO ESG mention in retail documents',
                    'message': 'ESG mentions PROHIBITED for Prospectus-limited retail documents',
                    'evidence': f"Found ESG content on {len(esg_slides)} slides. Elements: {', '.join(unique_elements)}",
                    'confidence': 90,
                    'method': 'AI_ENHANCED',
                    'ai_reasoning': f'Detected ESG content in retail document for prospectus-limited fund'
                }

        # ESG_005: Other funds (only baseline exclusions)
        if 'other' in classification_lower or 'autres' in classification_lower:
            doc_text = extract_all_text_from_doc(document)
            result = check_esg_baseline_with_ai(doc_text, ai_engine)

            if result.get('beyond_baseline') and result.get('confidence', 0) > 70:
                return {
                    'type': 'ESG',
                    'severity': 'MAJOR',
                    'slide': 'Document-wide',
                    'location': 'ESG sections',
                    'rule': 'ESG_005: Other funds → Only OBAM baseline exclusions allowed',
                    'message': 'ESG content goes beyond baseline exclusions (prohibited for Other funds)',
                    'evidence': f"Topics beyond baseline: {', '.join(result.get('esg_topics_found', [])[:5])}. {result.get('reasoning', '')} Confidence: {result.get('confidence', 0)}%",
                    'confidence': result.get('confidence', 75),
                    'method': 'AI_ENHANCED',
                    'ai_reasoning': result.get('reasoning', 'ESG content exceeds baseline exclusions')
                }

        # No violations found
        logger.info("ESG content distribution compliant")
        return None

    except Exception as e:
        logger.error(f"Error checking content distribution: {e}")
        return None


# ============================================================================
# SFDR COMPLIANCE TOOL
# ============================================================================

@tool
def check_sfdr_compliance(
    document: dict,
    esg_classification: str,
    metadata: dict,
    ai_engine=None
) -> Optional[dict]:
    """
    Check SFDR (Sustainable Finance Disclosure Regulation) compliance.

    Validates that SFDR classification mentions are consistent with fund's
    ESG approach and that SFDR-related content is appropriate.

    SFDR Articles:
    - Article 6: Minimal ESG integration
    - Article 8: Promotes environmental or social characteristics
    - Article 9: Has sustainable investment as objective

    Args:
        document: Document dictionary
        esg_classification: Fund's ESG classification
        metadata: Document metadata including SFDR classification
        ai_engine: AI engine instance for content analysis

    Returns:
        Violation dictionary if SFDR compliance is violated, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        doc_text = extract_all_text_from_doc(document).lower()
        sfdr_classification = metadata.get('sfdr_classification', '').lower()
        classification_lower = esg_classification.lower()

        # Check for SFDR mentions
        sfdr_keywords = ['sfdr', 'article 6', 'article 8', 'article 9', 'article six', 'article eight', 'article nine']
        has_sfdr_mention = any(keyword in doc_text for keyword in sfdr_keywords)

        if not has_sfdr_mention:
            logger.info("No SFDR mentions found in document")
            return None

        # Validate SFDR consistency with ESG classification
        # Article 9 funds should typically be "Engaging"
        if 'article 9' in doc_text or 'article nine' in doc_text:
            if 'engaging' not in classification_lower and 'engageante' not in classification_lower:
                logger.warning("Article 9 SFDR fund without Engaging ESG classification")
                return {
                    'type': 'ESG',
                    'severity': 'MAJOR',
                    'slide': 'Document-wide',
                    'location': 'SFDR mentions',
                    'rule': 'SFDR_CONSISTENCY: SFDR Article 9 should align with Engaging ESG approach',
                    'message': 'SFDR Article 9 classification inconsistent with ESG approach',
                    'evidence': f"Document mentions Article 9 SFDR but fund is classified as '{esg_classification}' (expected: Engaging)",
                    'confidence': 80,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Article 9 funds typically require Engaging ESG approach'
                }

        # Prospectus-limited funds should not have detailed SFDR content
        if 'prospectus' in classification_lower or 'limitée' in classification_lower or 'limited' in classification_lower:
            # Count SFDR-related content
            sfdr_content_indicators = [
                'sustainable investment',
                'environmental characteristics',
                'social characteristics',
                'taxonomy regulation',
                'principal adverse impacts',
                'pai',
                'do no significant harm',
                'dnsh'
            ]

            sfdr_detail_count = sum(1 for indicator in sfdr_content_indicators if indicator in doc_text)

            if sfdr_detail_count >= 3:
                return {
                    'type': 'ESG',
                    'severity': 'MAJOR',
                    'slide': 'Document-wide',
                    'location': 'SFDR content',
                    'rule': 'ESG_004: Prospectus-limited approach → Minimal SFDR content in retail documents',
                    'message': 'Excessive SFDR detail for Prospectus-limited fund',
                    'evidence': f"Found {sfdr_detail_count} detailed SFDR concepts in document (should be minimal for prospectus-limited funds)",
                    'confidence': 75,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Prospectus-limited funds should have minimal SFDR detail'
                }

        logger.info("SFDR compliance check passed")
        return None

    except Exception as e:
        logger.error(f"Error checking SFDR compliance: {e}")
        return None


# ============================================================================
# ESG TERMINOLOGY VALIDATION TOOL
# ============================================================================

@tool
def validate_esg_terminology(
    document: dict,
    esg_classification: str,
    client_type: str
) -> Optional[dict]:
    """
    Validate ESG terminology usage for accuracy and appropriateness.

    Checks for:
    - Misuse of ESG technical terms
    - Inappropriate ESG claims
    - Greenwashing indicators
    - Inconsistent ESG terminology

    Args:
        document: Document dictionary
        esg_classification: Fund's ESG classification
        client_type: Client type (retail, professional)

    Returns:
        Violation dictionary if terminology issues found, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        doc_text = extract_all_text_from_doc(document).lower()
        classification_lower = esg_classification.lower()

        # Professional funds are exempt
        if client_type.lower() == 'professional':
            return None

        # Define problematic terminology patterns
        greenwashing_terms = [
            'green fund',
            'sustainable fund',
            'esg fund',
            'responsible fund',
            'impact fund',
            'climate fund'
        ]

        # Check for greenwashing terms in non-engaging funds
        if 'engaging' not in classification_lower and 'engageante' not in classification_lower:
            found_greenwashing = [term for term in greenwashing_terms if term in doc_text]

            if found_greenwashing:
                return {
                    'type': 'ESG',
                    'severity': 'MAJOR',
                    'slide': 'Document-wide',
                    'location': 'ESG terminology',
                    'rule': 'ESG_TERMINOLOGY: ESG fund labels require Engaging classification',
                    'message': 'Inappropriate ESG fund labeling for non-Engaging fund',
                    'evidence': f"Found ESG fund labels: {', '.join(found_greenwashing[:3])} but fund is '{esg_classification}' (not Engaging)",
                    'confidence': 85,
                    'method': 'RULE_BASED',
                    'rule_hints': 'ESG fund labels should only be used for Engaging approach funds'
                }

        # Check for Article 9 claims without proper classification
        if ('article 9' in doc_text or 'article nine' in doc_text):
            if 'engaging' not in classification_lower and 'engageante' not in classification_lower:
                return {
                    'type': 'ESG',
                    'severity': 'CRITICAL',
                    'slide': 'Document-wide',
                    'location': 'SFDR claims',
                    'rule': 'SFDR_TERMINOLOGY: Article 9 claims require Engaging ESG approach',
                    'message': 'Article 9 SFDR claim without Engaging classification',
                    'evidence': f"Document claims Article 9 SFDR status but fund is '{esg_classification}' (expected: Engaging)",
                    'confidence': 90,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Article 9 funds must have Engaging ESG approach'
                }

        # Check for impact claims in non-engaging funds
        impact_terms = ['impact investing', 'measurable impact', 'impact measurement', 'impact reporting']
        found_impact = [term for term in impact_terms if term in doc_text]

        if found_impact and 'engaging' not in classification_lower and 'engageante' not in classification_lower:
            return {
                'type': 'ESG',
                'severity': 'MAJOR',
                'slide': 'Document-wide',
                'location': 'Impact claims',
                'rule': 'ESG_TERMINOLOGY: Impact claims require Engaging classification',
                'message': 'Impact investing claims inappropriate for non-Engaging fund',
                'evidence': f"Found impact claims: {', '.join(found_impact[:2])} but fund is '{esg_classification}' (not Engaging)",
                'confidence': 80,
                'method': 'RULE_BASED',
                'rule_hints': 'Impact investing terminology should be reserved for Engaging funds'
            }

        logger.info("ESG terminology validation passed")
        return None

    except Exception as e:
        logger.error(f"Error validating ESG terminology: {e}")
        return None


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# Export all tools for agent use
ESG_TOOLS = [
    check_esg_classification,
    check_content_distribution,
    check_sfdr_compliance,
    validate_esg_terminology
]


if __name__ == "__main__":
    # Test the tools
    logger.info("ESG Tools Module")
    logger.info("=" * 70)
    logger.info(f"Available tools: {len(ESG_TOOLS)}")
    for tool in ESG_TOOLS:
        logger.info(f"  - {tool.name}: {tool.description[:80]}...")
