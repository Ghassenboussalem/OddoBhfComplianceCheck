#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Evidence Extraction Tools for Multi-Agent Compliance System

This module provides tools for extracting and tracking evidence supporting compliance findings:
- Extract evidence for violations
- Find actual performance data (numbers with percentages)
- Find disclaimers using semantic matching
- Track locations and quotes
- Extract relevant text passages

Key Features:
- Performance data detection with context analysis
- Semantic disclaimer matching
- Location tracking within documents
- Quote extraction with context
- AI-enhanced evidence validation

Requirements: 2.3, 7.2, 7.4, 7.5, 9.3, 9.4
"""

import json
import re
import logging
import sys
import os
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tool_registry import tool, ToolCategory
from data_models import Evidence, PerformanceData, DisclaimerMatch

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# EVIDENCE EXTRACTION TOOL
# ============================================================================

@tool(
    name="extract_evidence",
    category=ToolCategory.ANALYSIS,
    description="Extract specific evidence supporting a compliance violation",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def extract_evidence(text: str, violation_type: str, location: str = "") -> Evidence:
    """
    Extract specific evidence for a compliance violation

    Extracts relevant quotes, locations, and context based on violation type:
    - performance_data: Finds actual performance numbers
    - missing_disclaimer: Extracts context where disclaimer should be
    - prohibited_phrase: Extracts phrases that triggered violation
    - Generic: Extracts relevant text passages

    Args:
        text: Text to analyze
        violation_type: Type of violation (e.g., "performance_data", "missing_disclaimer")
        location: Location in document (e.g., "Slide 2", "Cover Page")

    Returns:
        Evidence object with quotes, locations, context, and confidence

    Examples:
        >>> extract_evidence("Le fonds a généré +15.5% en 2023", "performance_data", "Slide 2")
        Evidence(quotes=["+15.5% - Le fonds a généré +15.5% en 2023"],
                 locations=["Slide 2"],
                 context="Found 1 performance data points",
                 confidence=90)

    Requirements: 2.3, 7.2, 7.4, 7.5, 9.3, 9.4
    """
    try:
        quotes = []
        locations = []
        context = ""
        confidence = 0

        # Extract evidence based on violation type
        if violation_type == "performance_data":
            # Find actual performance data
            perf_data_result = find_performance_data(text)
            perf_data = perf_data_result.result if hasattr(perf_data_result, 'result') else perf_data_result

            if perf_data:
                quotes = [f"{pd.value} - {pd.context[:100]}" for pd in perf_data[:3]]
                locations = [pd.location for pd in perf_data[:3]]
                context = f"Found {len(perf_data)} performance data points"
                confidence = 90

        elif violation_type == "missing_disclaimer":
            # Extract context around where disclaimer should be
            quotes = _extract_relevant_quotes(text, max_quotes=3)
            locations = [location] if location else []
            context = "Performance data present without required disclaimer"
            confidence = 85

        elif violation_type == "prohibited_phrase":
            # Extract phrases that triggered the violation
            quotes = _extract_relevant_quotes(text, max_quotes=5)
            locations = [location] if location else []
            context = "Potentially prohibited phrases detected"
            confidence = 70

        else:
            # Generic evidence extraction
            quotes = _extract_relevant_quotes(text, max_quotes=3)
            locations = [location] if location else []
            context = f"Evidence for {violation_type}"
            confidence = 60

        return Evidence(
            quotes=quotes,
            locations=locations,
            context=context,
            confidence=confidence
        )

    except Exception as e:
        logger.error(f"Evidence extraction failed: {e}")
        return Evidence(
            quotes=[],
            locations=[],
            context=f"Error extracting evidence: {str(e)}",
            confidence=0
        )


# ============================================================================
# PERFORMANCE DATA DETECTION TOOL
# ============================================================================

@tool(
    name="find_performance_data",
    category=ToolCategory.ANALYSIS,
    description="Find actual performance numbers (percentages) in text",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def find_performance_data(text: str, ai_engine=None) -> List[PerformanceData]:
    """
    Find actual performance numbers in text

    Detects:
    - Percentage values with +/- signs: +15.5%, -3.2%
    - Performance/return/rendement with numbers
    - Chart data with percentages
    - Distinguishes actual data from descriptive text

    Args:
        text: Text to analyze
        ai_engine: Optional AIEngine for semantic analysis

    Returns:
        List of PerformanceData objects with value, context, location, and confidence

    Examples:
        >>> find_performance_data("Le fonds a généré +15.5% en 2023")
        [PerformanceData(value="+15.5%", context="Le fonds a généré +15.5% en 2023",
                         location="Unknown Location", confidence=75)]

        >>> find_performance_data("L'objectif de performance attractive")
        []  # Descriptive text, not actual data

    Requirements: 2.3, 7.2, 7.4, 7.5, 9.3, 9.4
    """
    performance_data = []

    try:
        # Convert to string if dict/json
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)

        text_lower = text.lower()

        # Performance data patterns
        performance_patterns = [
            r'[+\-]?\d+[.,]\d+\s*%',  # +15.5%, -3.2%
            r'[+\-]?\d+\s*%',  # +15%, -3%
            r'\d+[.,]\d+\s*%',  # 15.5%
            r'performance.*?[+\-]?\d+[.,]?\d*\s*%',  # performance of 15%
            r'return.*?[+\-]?\d+[.,]?\d*\s*%',  # return of 15%
            r'rendement.*?[+\-]?\d+[.,]?\d*\s*%',  # rendement de 15%
        ]

        # Check if this is descriptive text without actual data
        descriptive_phrases = [
            'performance objective',
            'objectif de performance',
            'attractive performance',
            'performance attractive',
            'strong performance',
            'performance solide',
            'performance potential',
            'potentiel de performance',
        ]

        # If only descriptive phrases, not actual data
        has_descriptive_only = any(phrase in text_lower for phrase in descriptive_phrases)

        # Search for performance patterns
        for pattern in performance_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                value = match.group(0)
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                context = text[start_pos:end_pos].strip()

                # Check if this is in a descriptive context
                context_lower = context.lower()
                is_descriptive = any(phrase in context_lower for phrase in descriptive_phrases)

                # Skip if only descriptive and no actual numbers
                if has_descriptive_only and is_descriptive and not re.search(r'\d{4}', context):
                    continue

                # Determine location from context
                location = _extract_location_from_context(text, match.start())

                # Calculate confidence based on context
                confidence = _calculate_performance_confidence(context, value)

                performance_data.append(PerformanceData(
                    value=value,
                    context=context,
                    location=location,
                    confidence=confidence
                ))

        # Use AI for semantic analysis if available
        if ai_engine and performance_data:
            performance_data = _enhance_with_ai_analysis(text, performance_data, ai_engine)

        # Remove duplicates and sort by confidence
        performance_data = _deduplicate_performance_data(performance_data)
        performance_data.sort(key=lambda x: x.confidence, reverse=True)

        logger.debug(f"Found {len(performance_data)} performance data points")
        return performance_data

    except Exception as e:
        logger.error(f"Performance data extraction failed: {e}")
        return []


# ============================================================================
# DISCLAIMER DETECTION TOOL
# ============================================================================

@tool(
    name="find_disclaimer",
    category=ToolCategory.ANALYSIS,
    description="Find disclaimer using semantic similarity matching",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=True,
    max_retries=2
)
def find_disclaimer(text: str, required_disclaimer: str, ai_engine=None) -> DisclaimerMatch:
    """
    Find disclaimer using semantic similarity

    Matches variations and paraphrases of required disclaimers:
    - "Les performances passées ne préjugent pas..."
    - "Past performance is not indicative..."
    - Variations and paraphrases

    Uses both keyword matching and optional AI semantic analysis.

    Args:
        text: Text to search
        required_disclaimer: Required disclaimer text
        ai_engine: Optional AIEngine for semantic matching

    Returns:
        DisclaimerMatch with found status, text, location, similarity score, and confidence

    Examples:
        >>> find_disclaimer("Les performances passées ne préjugent pas des performances futures",
        ...                 "Les performances passées ne préjugent pas des performances futures")
        DisclaimerMatch(found=True, text="Les performances passées...",
                       location="Unknown Location", similarity_score=100, confidence=95)

    Requirements: 2.3, 7.2, 7.4, 7.5, 9.3, 9.4
    """
    try:
        # Convert to string if dict/json
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)

        text_lower = text.lower()
        required_lower = required_disclaimer.lower()

        # Check for exact or near-exact match
        if required_lower in text_lower:
            location = _extract_location_from_context(text, text_lower.index(required_lower))
            return DisclaimerMatch(
                found=True,
                text=required_disclaimer,
                location=location,
                similarity_score=100,
                confidence=95
            )

        # Disclaimer patterns (French and English)
        disclaimer_keywords = [
            'performances passées',
            'past performance',
            'ne préjugent pas',
            'not indicative',
            'ne garantit pas',
            'does not guarantee',
            'no guarantee',
            'aucune garantie',
        ]

        # Check for keyword-based matching
        disclaimer_score = 0
        matched_keywords = []

        for keyword in disclaimer_keywords:
            if keyword.lower() in text_lower:
                disclaimer_score += 1
                matched_keywords.append(keyword)

        # If we found multiple disclaimer keywords, likely a match
        if disclaimer_score >= 2:
            # Extract the disclaimer text
            disclaimer_text = _extract_disclaimer_text(text, matched_keywords)
            location = _extract_location_from_context(text, 0)

            similarity_score = min(100, disclaimer_score * 30)
            confidence = min(90, disclaimer_score * 25)

            return DisclaimerMatch(
                found=True,
                text=disclaimer_text,
                location=location,
                similarity_score=similarity_score,
                confidence=confidence
            )

        # Use AI for semantic matching if available
        if ai_engine:
            ai_match = _ai_disclaimer_match(text, required_disclaimer, ai_engine)
            if ai_match:
                return ai_match

        # No disclaimer found
        return DisclaimerMatch(
            found=False,
            text="",
            location="",
            similarity_score=0,
            confidence=0
        )

    except Exception as e:
        logger.error(f"Disclaimer search failed: {e}")
        return DisclaimerMatch(
            found=False,
            text="",
            location="",
            similarity_score=0,
            confidence=0
        )


# ============================================================================
# LOCATION TRACKING TOOL
# ============================================================================

@tool(
    name="track_location",
    category=ToolCategory.UTILITY,
    description="Extract location information from text context",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=False
)
def track_location(text: str, position: int = 0) -> str:
    """
    Extract location information from text context

    Looks for slide markers, page indicators, and section names in nearby text.

    Args:
        text: Full text
        position: Position in text to search around

    Returns:
        Location string (e.g., "Slide 2", "Cover Page", "Disclaimer Page")

    Examples:
        >>> track_location('{"slide": "2", "content": "..."}', 0)
        "Slide 2"

        >>> track_location("page_de_garde: {...}", 0)
        "Cover Page"

    Requirements: 7.2, 7.4, 9.4
    """
    return _extract_location_from_context(text, position)


# ============================================================================
# QUOTE EXTRACTION TOOL
# ============================================================================

@tool(
    name="extract_quotes",
    category=ToolCategory.UTILITY,
    description="Extract relevant quotes from text",
    cache_enabled=True,
    cache_ttl_seconds=3600,
    retry_enabled=False
)
def extract_quotes(text: str, max_quotes: int = 3, max_length: int = 150) -> List[str]:
    """
    Extract relevant quotes from text

    Splits text into sentences and extracts the most relevant ones.
    Filters out very short or very long sentences.

    Args:
        text: Text to extract from
        max_quotes: Maximum number of quotes to extract
        max_length: Maximum length per quote

    Returns:
        List of quote strings

    Examples:
        >>> extract_quotes("First sentence. Second sentence. Third sentence.", max_quotes=2)
        ["First sentence", "Second sentence"]

    Requirements: 7.2, 7.4, 9.4
    """
    return _extract_relevant_quotes(text, max_quotes, max_length)


# ============================================================================
# PRIVATE HELPER FUNCTIONS
# ============================================================================

def _extract_relevant_quotes(text: str, max_quotes: int = 3, max_length: int = 150) -> List[str]:
    """
    Extract relevant quotes from text

    Args:
        text: Text to extract from
        max_quotes: Maximum number of quotes
        max_length: Maximum length per quote

    Returns:
        List of quote strings
    """
    try:
        # Convert to string if dict/json
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        # Filter out very short or very long sentences
        sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 300]

        # Take first max_quotes sentences
        quotes = sentences[:max_quotes]

        # Truncate if needed
        quotes = [q[:max_length] + "..." if len(q) > max_length else q for q in quotes]

        return quotes

    except Exception as e:
        logger.error(f"Quote extraction failed: {e}")
        return []


def _extract_location_from_context(text: str, position: int) -> str:
    """
    Extract location information from text context

    Args:
        text: Full text
        position: Position in text

    Returns:
        Location string (e.g., "Slide 2", "Cover Page")
    """
    try:
        # Look for slide markers in nearby text
        context_start = max(0, position - 500)
        context_end = min(len(text), position + 500)
        context = text[context_start:context_end]

        # Check for slide indicators
        slide_patterns = [
            r'slide[_\s]*(\d+)',
            r'page[_\s]*(\d+)',
            r'diapositive[_\s]*(\d+)',
            r'"slide":\s*"([^"]+)"',
            r'"page":\s*"([^"]+)"',
        ]

        for pattern in slide_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return f"Slide {match.group(1)}"

        # Check for special pages
        if 'cover' in context.lower() or 'garde' in context.lower():
            return "Cover Page"
        if 'disclaimer' in context.lower() or 'avertissement' in context.lower():
            return "Disclaimer Page"

        return "Unknown Location"

    except Exception as e:
        logger.error(f"Location extraction failed: {e}")
        return "Unknown Location"


def _calculate_performance_confidence(context: str, value: str) -> int:
    """
    Calculate confidence score for performance data

    Args:
        context: Context around the value
        value: Performance value

    Returns:
        Confidence score (0-100)
    """
    confidence = 50  # Base confidence

    context_lower = context.lower()

    # Increase confidence for clear performance indicators
    performance_indicators = [
        'performance', 'return', 'rendement', 'gain', 'profit',
        'yield', 'growth', 'croissance', 'résultat'
    ]

    for indicator in performance_indicators:
        if indicator in context_lower:
            confidence += 10
            break

    # Increase confidence for year references
    if re.search(r'20\d{2}', context):
        confidence += 15

    # Increase confidence for clear percentage format
    if re.match(r'[+\-]\d+[.,]\d+\s*%', value):
        confidence += 10

    # Decrease confidence for descriptive contexts
    descriptive_phrases = ['objective', 'objectif', 'potential', 'potentiel', 'aim', 'vise']
    if any(phrase in context_lower for phrase in descriptive_phrases):
        confidence -= 20

    return max(0, min(100, confidence))


def _extract_disclaimer_text(text: str, matched_keywords: List[str]) -> str:
    """
    Extract the full disclaimer text based on matched keywords

    Args:
        text: Full text
        matched_keywords: Keywords that were matched

    Returns:
        Extracted disclaimer text
    """
    try:
        text_lower = text.lower()

        # Find position of first matched keyword
        first_pos = len(text)
        for keyword in matched_keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1 and pos < first_pos:
                first_pos = pos

        if first_pos < len(text):
            # Extract sentence containing the keyword
            start = max(0, first_pos - 50)
            end = min(len(text), first_pos + 200)

            # Find sentence boundaries
            excerpt = text[start:end]

            # Clean up
            excerpt = excerpt.strip()
            if not excerpt[0].isupper():
                # Find start of sentence
                sentence_start = excerpt.find('. ')
                if sentence_start != -1:
                    excerpt = excerpt[sentence_start + 2:]

            return excerpt[:200] + "..." if len(excerpt) > 200 else excerpt

        return "Disclaimer text found"

    except Exception as e:
        logger.error(f"Disclaimer text extraction failed: {e}")
        return "Disclaimer present"


def _deduplicate_performance_data(performance_data: List[PerformanceData]) -> List[PerformanceData]:
    """
    Remove duplicate performance data entries

    Args:
        performance_data: List of PerformanceData objects

    Returns:
        Deduplicated list
    """
    seen = set()
    unique_data = []

    for pd in performance_data:
        # Create a key based on value and location
        key = (pd.value, pd.location)
        if key not in seen:
            seen.add(key)
            unique_data.append(pd)

    return unique_data


def _enhance_with_ai_analysis(text: str, performance_data: List[PerformanceData], ai_engine) -> List[PerformanceData]:
    """
    Enhance performance data with AI semantic analysis

    Args:
        text: Full text
        performance_data: List of PerformanceData objects
        ai_engine: AIEngine instance

    Returns:
        Enhanced list with updated confidence scores
    """
    try:
        if not ai_engine:
            return performance_data

        # Create prompt for AI analysis
        prompt = f"""Analyze these performance data points and determine if they represent actual historical performance data or just descriptive text:

TEXT: {text[:1000]}

PERFORMANCE DATA FOUND:
{json.dumps([{'value': pd.value, 'context': pd.context[:100]} for pd in performance_data[:5]], indent=2)}

For each data point, determine:
1. Is this actual historical performance data? (true/false)
2. Confidence level (0-100)

Respond with JSON:
{{
  "analysis": [
    {{"index": 0, "is_actual_data": true/false, "confidence": 0-100, "reasoning": "explanation"}},
    ...
  ]
}}"""

        response = ai_engine.call_with_cache(
            prompt=prompt,
            system_message="You are a financial document analyzer. Respond only with valid JSON."
        )

        if response and response.parsed_json and 'analysis' in response.parsed_json:
            analysis = response.parsed_json['analysis']

            for i, item in enumerate(analysis):
                if i < len(performance_data):
                    if not item.get('is_actual_data', True):
                        # Reduce confidence if AI says it's not actual data
                        performance_data[i].confidence = min(
                            performance_data[i].confidence,
                            item.get('confidence', 50)
                        )
                    else:
                        # Boost confidence if AI confirms it's actual data
                        performance_data[i].confidence = max(
                            performance_data[i].confidence,
                            item.get('confidence', 70)
                        )

        return performance_data

    except Exception as e:
        logger.error(f"AI enhancement failed: {e}")
        return performance_data


def _ai_disclaimer_match(text: str, required_disclaimer: str, ai_engine) -> Optional[DisclaimerMatch]:
    """
    Use AI to semantically match disclaimers

    Args:
        text: Text to search
        required_disclaimer: Required disclaimer
        ai_engine: AIEngine instance

    Returns:
        DisclaimerMatch if AI finds a match
    """
    try:
        if not ai_engine:
            return None

        prompt = f"""Does this text contain a disclaimer equivalent to the required disclaimer?

REQUIRED DISCLAIMER: {required_disclaimer}

TEXT TO SEARCH: {text[:2000]}

Consider semantic equivalence, not just exact wording. Look for:
- Past performance disclaimers
- No guarantee statements
- Risk warnings

Respond with JSON:
{{
  "found": true/false,
  "disclaimer_text": "extracted text if found",
  "similarity_score": 0-100,
  "confidence": 0-100,
  "reasoning": "explanation"
}}"""

        response = ai_engine.call_with_cache(
            prompt=prompt,
            system_message="You are a compliance expert. Respond only with valid JSON."
        )

        if response and response.parsed_json:
            result = response.parsed_json

            if result.get('found', False):
                location = _extract_location_from_context(text, 0)

                return DisclaimerMatch(
                    found=True,
                    text=result.get('disclaimer_text', 'Disclaimer found'),
                    location=location,
                    similarity_score=result.get('similarity_score', 70),
                    confidence=result.get('confidence', 70)
                )

        return None

    except Exception as e:
        logger.error(f"AI disclaimer matching failed: {e}")
        return None


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all evidence extraction tools for easy import
EVIDENCE_TOOLS = [
    extract_evidence,
    find_performance_data,
    find_disclaimer,
    track_location,
    extract_quotes
]


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Evidence Extraction Tools - Standalone Test")
    logger.info("="*70)

    # Test performance data extraction
    test_text_perf = """
    Le fonds a généré une performance de +15.5% en 2023.
    Le rendement annualisé est de 8.2% sur 5 ans.
    L'objectif de performance est d'obtenir des résultats attractifs.
    """

    logger.info("\n🔍 Testing find_performance_data...")
    perf_result = find_performance_data(test_text_perf)
    perf_data = perf_result.result if hasattr(perf_result, 'result') else perf_result
    logger.info(f"  Found {len(perf_data)} performance data points:")
    for pd in perf_data:
        logger.info(f"    - {pd.value} (confidence: {pd.confidence}%)")
        logger.info(f"      Context: {pd.context[:80]}...")

    # Test disclaimer detection
    disclaimer_text = """
    Les performances passées ne préjugent pas des performances futures.
    Aucune garantie ne peut être donnée quant aux résultats futurs.
    """

    logger.info("\n🔍 Testing find_disclaimer...")
    disclaimer_result = find_disclaimer(
        disclaimer_text,
        "Les performances passées ne préjugent pas des performances futures"
    )
    disclaimer = disclaimer_result.result if hasattr(disclaimer_result, 'result') else disclaimer_result
    logger.info(f"  Disclaimer found: {disclaimer.found}")
    if disclaimer.found:
        logger.info(f"  Similarity score: {disclaimer.similarity_score}%")
        logger.info(f"  Confidence: {disclaimer.confidence}%")
        logger.info(f"  Text: {disclaimer.text[:100]}...")

    # Test evidence extraction
    logger.info("\n🔍 Testing extract_evidence...")
    evidence_result = extract_evidence(
        test_text_perf,
        "performance_data",
        "Slide 2"
    )
    evidence = evidence_result.result if hasattr(evidence_result, 'result') else evidence_result
    logger.info(f"  Quotes: {len(evidence.quotes)}")
    for quote in evidence.quotes:
        logger.info(f"    - {quote[:80]}...")
    logger.info(f"  Locations: {evidence.locations}")
    logger.info(f"  Confidence: {evidence.confidence}%")

    # Test location tracking
    logger.info("\n🔍 Testing track_location...")
    location_result = track_location('{"slide": "2", "content": "test"}', 0)
    location = location_result.result if hasattr(location_result, 'result') else location_result
    logger.info(f"  Location: {location}")

    # Test quote extraction
    logger.info("\n🔍 Testing extract_quotes...")
    quotes_result = extract_quotes(test_text_perf, max_quotes=2)
    quotes = quotes_result.result if hasattr(quotes_result, 'result') else quotes_result
    logger.info(f"  Extracted {len(quotes)} quotes:")
    for quote in quotes:
        logger.info(f"    - {quote}")

    logger.info("\n" + "="*70)
    logger.info("All evidence extraction tools tested!")
    logger.info("="*70)
