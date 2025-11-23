#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Example Tool Definitions

This module demonstrates how to define tools for the multi-agent compliance system
using the tool registry framework.

These examples show the patterns that will be used when migrating existing
compliance checking functions to the tool-based architecture.
"""

from typing import Optional, Set, Dict, Any
from tools.tool_registry import tool, ToolCategory


# ============================================================================
# PREPROCESSING TOOLS
# ============================================================================

@tool(
    name="extract_metadata",
    category=ToolCategory.PREPROCESSING,
    description="Extract metadata from document including fund ISIN, client type, and document type",
    cache_enabled=True,
    cache_ttl_seconds=3600
)
def extract_metadata(document: dict) -> dict:
    """
    Extract metadata from document

    Args:
        document: Document dictionary with document_metadata field

    Returns:
        Dictionary containing extracted metadata
    """
    metadata = document.get("document_metadata", {})
    return {
        "fund_isin": metadata.get("fund_isin"),
        "client_type": metadata.get("client_type", "retail"),
        "document_type": metadata.get("document_type", "fund_presentation"),
        "fund_name": metadata.get("fund_name"),
        "esg_classification": metadata.get("fund_esg_classification", "other")
    }


@tool(
    name="normalize_document",
    category=ToolCategory.PREPROCESSING,
    description="Normalize document structure to ensure all expected fields exist",
    cache_enabled=True,
    cache_ttl_seconds=3600
)
def normalize_document(document: dict) -> dict:
    """
    Normalize document structure

    Args:
        document: Raw document dictionary

    Returns:
        Normalized document with all expected fields
    """
    return {
        "page_de_garde": document.get("page_de_garde", {}),
        "slide_2": document.get("slide_2", {}),
        "pages_suivantes": document.get("pages_suivantes", []),
        "page_de_fin": document.get("page_de_fin", {}),
        "document_metadata": document.get("document_metadata", {})
    }


# ============================================================================
# CHECKING TOOLS
# ============================================================================

@tool(
    name="check_promotional_mention",
    category=ToolCategory.CHECKING,
    description="Check for promotional document mention on cover page",
    retry_enabled=True,
    max_retries=2,
    cache_enabled=True
)
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """
    Check if promotional document mention exists on cover page

    Args:
        document: Normalized document
        config: Configuration dictionary

    Returns:
        Violation dictionary if check fails, None otherwise
    """
    page_de_garde = document.get("page_de_garde", {})

    # Check if promotional mention exists
    if not page_de_garde.get("promotional_mention"):
        return {
            "type": "STRUCTURE",
            "rule": "promotional_mention",
            "severity": "high",
            "slide": "page_de_garde",
            "location": "cover_page",
            "evidence": "No promotional document mention found",
            "confidence": 95,
            "ai_reasoning": "Cover page must contain promotional document mention"
        }

    return None


@tool(
    name="check_target_audience",
    category=ToolCategory.CHECKING,
    description="Check target audience specification based on client type",
    retry_enabled=True,
    cache_enabled=True
)
def check_target_audience(document: dict, client_type: str) -> Optional[dict]:
    """
    Check if target audience is properly specified

    Args:
        document: Normalized document
        client_type: Client type (retail or professional)

    Returns:
        Violation dictionary if check fails, None otherwise
    """
    page_de_garde = document.get("page_de_garde", {})
    target_audience = page_de_garde.get("target_audience", "")

    # Check based on client type
    if client_type == "retail" and "retail" not in target_audience.lower():
        return {
            "type": "STRUCTURE",
            "rule": "target_audience",
            "severity": "high",
            "slide": "page_de_garde",
            "location": "cover_page",
            "evidence": f"Target audience: {target_audience}",
            "confidence": 90,
            "ai_reasoning": "Retail documents must specify retail target audience"
        }

    return None


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

@tool(
    name="analyze_context",
    category=ToolCategory.ANALYSIS,
    description="Analyze text context to determine WHO performs actions and WHAT the intent is",
    retry_enabled=True,
    max_retries=3,
    cache_enabled=True,
    cache_ttl_seconds=1800
)
def analyze_context(text: str, check_type: str) -> dict:
    """
    Analyze text context using semantic understanding

    Args:
        text: Text to analyze
        check_type: Type of compliance check

    Returns:
        Dictionary with context analysis results
    """
    # This is a simplified example - real implementation would use AI
    text_lower = text.lower()

    return {
        "is_fund_description": "fund" in text_lower and "strategy" in text_lower,
        "is_client_advice": "should" in text_lower or "recommend" in text_lower,
        "is_factual_statement": "is" in text_lower or "has" in text_lower,
        "subject": "fund" if "fund" in text_lower else "unknown",
        "confidence": 85
    }


@tool(
    name="classify_intent",
    category=ToolCategory.ANALYSIS,
    description="Classify text intent as ADVICE, DESCRIPTION, FACT, or EXAMPLE",
    retry_enabled=True,
    cache_enabled=True,
    cache_ttl_seconds=1800
)
def classify_intent(text: str) -> dict:
    """
    Classify the intent of text

    Args:
        text: Text to classify

    Returns:
        Dictionary with intent classification
    """
    # Simplified example - real implementation would use AI
    text_lower = text.lower()

    if any(word in text_lower for word in ["should", "recommend", "advise"]):
        intent = "ADVICE"
        confidence = 90
    elif any(word in text_lower for word in ["for example", "such as", "like"]):
        intent = "EXAMPLE"
        confidence = 85
    elif any(word in text_lower for word in ["strategy", "objective", "approach"]):
        intent = "DESCRIPTION"
        confidence = 80
    else:
        intent = "FACT"
        confidence = 70

    return {
        "intent": intent,
        "confidence": confidence,
        "reasoning": f"Text classified as {intent} based on keywords"
    }


# ============================================================================
# REVIEW TOOLS
# ============================================================================

@tool(
    name="calculate_priority_score",
    category=ToolCategory.REVIEW,
    description="Calculate priority score for review queue based on confidence and severity",
    cache_enabled=False,  # Don't cache - scores may change
    retry_enabled=False
)
def calculate_priority_score(violation: dict) -> int:
    """
    Calculate priority score for a violation

    Args:
        violation: Violation dictionary

    Returns:
        Priority score (0-100, higher = more urgent)
    """
    confidence = violation.get("confidence", 100)
    severity = violation.get("severity", "medium")

    # Base score is inverse of confidence (lower confidence = higher priority)
    base_score = 100 - confidence

    # Adjust for severity
    severity_multiplier = {
        "critical": 1.5,
        "high": 1.2,
        "medium": 1.0,
        "low": 0.8
    }

    multiplier = severity_multiplier.get(severity, 1.0)
    priority_score = int(base_score * multiplier)

    return min(priority_score, 100)


@tool(
    name="filter_violations_by_confidence",
    category=ToolCategory.REVIEW,
    description="Filter violations based on confidence threshold",
    cache_enabled=False,
    retry_enabled=False
)
def filter_violations_by_confidence(
    violations: list,
    threshold: int = 70
) -> Dict[str, list]:
    """
    Filter violations by confidence threshold

    Args:
        violations: List of violation dictionaries
        threshold: Confidence threshold (0-100)

    Returns:
        Dictionary with 'high_confidence' and 'low_confidence' lists
    """
    high_confidence = []
    low_confidence = []

    for violation in violations:
        confidence = violation.get("confidence", 100)
        if confidence >= threshold:
            high_confidence.append(violation)
        else:
            low_confidence.append(violation)

    return {
        "high_confidence": high_confidence,
        "low_confidence": low_confidence,
        "threshold": threshold,
        "total": len(violations)
    }


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@tool(
    name="extract_text_from_slide",
    category=ToolCategory.UTILITY,
    description="Extract all text content from a specific slide",
    cache_enabled=True,
    retry_enabled=False
)
def extract_text_from_slide(document: dict, slide_name: str) -> str:
    """
    Extract text from a specific slide

    Args:
        document: Document dictionary
        slide_name: Name of the slide (e.g., 'page_de_garde', 'slide_2')

    Returns:
        Concatenated text from the slide
    """
    slide = document.get(slide_name, {})

    # Extract text from all fields
    text_parts = []
    for key, value in slide.items():
        if isinstance(value, str):
            text_parts.append(value)
        elif isinstance(value, list):
            text_parts.extend([str(item) for item in value if isinstance(item, str)])

    return " ".join(text_parts)


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    import json
    from tools.tool_registry import get_tool_stats, list_all_tools

    logger.info("=" * 70)
    logger.info("EXAMPLE TOOLS DEMONSTRATION")
    logger.info("=" * 70)

    # List all registered tools
    logger.info("\nRegistered Tools:")
    for tool_name in list_all_tools():
        logger.info(f"  - {tool_name}")

    # Test preprocessing tools
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING TOOLS")
    logger.info("=" * 70)

    test_doc = {
        "document_metadata": {
            "fund_isin": "FR0000123456",
            "client_type": "professional",
            "fund_name": "Test Fund"
        },
        "page_de_garde": {
            "title": "Fund Presentation"
        }
    }

    logger.info("\n1. Extract Metadata:")
    result = extract_metadata(test_doc)
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Result: {json.dumps(result.result, indent=2)}")
    logger.info(f"   Cached: {result.cached}")
    logger.info(f"   Time: {result.execution_time:.3f}s")

    logger.info("\n2. Normalize Document:")
    result = normalize_document(test_doc)
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Cached: {result.cached}")

    # Test checking tools
    logger.info("\n" + "=" * 70)
    logger.info("CHECKING TOOLS")
    logger.info("=" * 70)

    logger.info("\n3. Check Promotional Mention:")
    result = check_promotional_mention(test_doc, {})
    logger.info(f"   Success: {result.success}")
    if result.result:
        logger.info(f"   Violation: {result.result['rule']}")
        logger.info(f"   Confidence: {result.result['confidence']}")

    # Test analysis tools
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS TOOLS")
    logger.info("=" * 70)

    logger.info("\n4. Analyze Context:")
    result = analyze_context("The fund strategy focuses on growth", "SECURITIES")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Result: {json.dumps(result.result, indent=2)}")

    logger.info("\n5. Classify Intent:")
    result = classify_intent("You should invest in this fund")
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Intent: {result.result['intent']}")
    logger.info(f"   Confidence: {result.result['confidence']}")

    # Test review tools
    logger.info("\n" + "=" * 70)
    logger.info("REVIEW TOOLS")
    logger.info("=" * 70)

    test_violation = {
        "confidence": 65,
        "severity": "high"
    }

    logger.info("\n6. Calculate Priority Score:")
    result = calculate_priority_score(test_violation)
    logger.info(f"   Success: {result.success}")
    logger.info(f"   Priority Score: {result.result}")

    # Show statistics
    logger.info("\n" + "=" * 70)
    logger.info("TOOL STATISTICS")
    logger.info("=" * 70)

    stats = get_tool_stats()
    logger.info(f"\nCache Statistics:")
    logger.info(f"  Hits: {stats['cache']['hits']}")
    logger.info(f"  Misses: {stats['cache']['misses']}")
    logger.info(f"  Hit Rate: {stats['cache']['hit_rate_percent']:.1f}%")

    logger.info(f"\nTool Execution Statistics:")
    for tool_name, tool_stats in stats['tools'].items():
        if tool_stats['total_calls'] > 0:
            logger.info(f"\n  {tool_name}:")
            logger.info(f"    Total Calls: {tool_stats['total_calls']}")
            logger.info(f"    Success Rate: {tool_stats['successful_calls']}/{tool_stats['total_calls']}")
            logger.info(f"    Avg Time: {tool_stats['avg_execution_time']:.3f}s")
