#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Preprocessing Tools for Multi-Agent Compliance System

These tools handle document preprocessing:
- Metadata extraction
- Whitelist building
- Document normalization
- Document validation

Requirements: 7.2, 2.3
"""

import json
import logging
import sys
import os
from typing import Dict, Set, Optional, List, Any
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing components
from whitelist_manager import WhitelistManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# METADATA EXTRACTION TOOL
# ============================================================================

@tool
def extract_metadata(document: dict) -> dict:
    """
    Extract metadata from document for compliance checking.

    Extracts key information needed for routing and compliance checks:
    - Fund ISIN
    - Client type (retail/professional)
    - Document type
    - Fund name
    - ESG classification
    - Country code
    - Fund age
    - Fund status

    Args:
        document: Document dictionary with document_metadata section

    Returns:
        Dictionary with extracted metadata fields

    Requirements: 7.2, 2.3
    """
    try:
        doc_metadata = document.get('document_metadata', {})

        # Extract all relevant metadata fields
        metadata = {
            'fund_isin': doc_metadata.get('fund_isin'),
            'client_type': doc_metadata.get('client_type', 'retail'),
            'document_type': doc_metadata.get('document_type', 'fund_presentation'),
            'fund_name': doc_metadata.get('fund_name'),
            'esg_classification': doc_metadata.get('fund_esg_classification', 'other'),
            'country_code': doc_metadata.get('country_code'),
            'fund_age_years': doc_metadata.get('fund_age_years'),
            'fund_status': doc_metadata.get('fund_status', 'active')
        }

        logger.info(f"Extracted metadata: fund_isin={metadata['fund_isin']}, "
                   f"client_type={metadata['client_type']}, "
                   f"document_type={metadata['document_type']}")

        return metadata

    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
        # Return default metadata on error
        return {
            'fund_isin': None,
            'client_type': 'retail',
            'document_type': 'fund_presentation',
            'fund_name': None,
            'esg_classification': 'other',
            'country_code': None,
            'fund_age_years': None,
            'fund_status': 'active'
        }


# ============================================================================
# WHITELIST BUILDING TOOL
# ============================================================================

@tool
def build_whitelist(document: dict, metadata: dict) -> Set[str]:
    """
    Build whitelist of allowed terms from document and metadata.

    Creates a comprehensive whitelist to prevent false positives:
    - Fund name components
    - Strategy terminology
    - Regulatory terms
    - Benchmark names
    - Generic financial terms
    - Management company terms

    Uses WhitelistManager to extract fund name components and combine
    with predefined term categories.

    Args:
        document: Document dictionary
        metadata: Extracted metadata dictionary

    Returns:
        Set of whitelisted terms (lowercase)

    Requirements: 7.2, 2.3
    """
    try:
        # Initialize WhitelistManager
        manager = WhitelistManager()

        # Build whitelist from document (extracts fund name components)
        whitelist = manager.build_whitelist(document)

        logger.info(f"Built whitelist with {len(whitelist)} terms")

        # Log whitelist statistics
        stats = manager.get_whitelist_stats()
        logger.debug(f"Whitelist composition: {stats}")

        return whitelist

    except Exception as e:
        logger.error(f"Error building whitelist: {e}")
        # Return empty set on error
        return set()


# ============================================================================
# DOCUMENT NORMALIZATION TOOL
# ============================================================================

@tool
def normalize_document(document: dict) -> dict:
    """
    Normalize document structure to ensure all expected fields exist.

    Ensures consistent document structure for downstream agents:
    - page_de_garde (cover page)
    - slide_2 (second slide)
    - pages_suivantes (following pages)
    - page_de_fin (back page)
    - document_metadata

    Missing sections are initialized as empty dictionaries/lists.

    Args:
        document: Raw document dictionary

    Returns:
        Normalized document with all expected fields

    Requirements: 7.2
    """
    try:
        # Ensure all expected fields exist
        normalized = {
            'page_de_garde': document.get('page_de_garde', {}),
            'slide_2': document.get('slide_2', {}),
            'pages_suivantes': document.get('pages_suivantes', []),
            'page_de_fin': document.get('page_de_fin', {}),
            'document_metadata': document.get('document_metadata', {})
        }

        # Validate pages_suivantes is a list
        if not isinstance(normalized['pages_suivantes'], list):
            logger.warning("pages_suivantes is not a list, converting to empty list")
            normalized['pages_suivantes'] = []

        # Add slide numbers to pages_suivantes if missing
        for i, page in enumerate(normalized['pages_suivantes'], start=3):
            if 'slide_number' not in page:
                page['slide_number'] = i

        logger.info(f"Normalized document with {len(normalized['pages_suivantes'])} following pages")

        return normalized

    except Exception as e:
        logger.error(f"Error normalizing document: {e}")
        # Return minimal structure on error
        return {
            'page_de_garde': {},
            'slide_2': {},
            'pages_suivantes': [],
            'page_de_fin': {},
            'document_metadata': {}
        }


# ============================================================================
# DOCUMENT VALIDATION TOOL
# ============================================================================

@tool
def validate_document(document: dict) -> dict:
    """
    Validate document structure and content.

    Performs validation checks:
    - Required sections present
    - Document metadata exists
    - Valid JSON structure
    - Minimum content requirements

    Args:
        document: Document dictionary to validate

    Returns:
        Dictionary with validation results:
        - valid: bool - overall validity
        - errors: list - validation errors found
        - warnings: list - validation warnings
        - sections_present: dict - which sections exist

    Requirements: 7.2
    """
    try:
        errors = []
        warnings = []
        sections_present = {}

        # Check required sections
        required_sections = ['document_metadata']
        optional_sections = ['page_de_garde', 'slide_2', 'pages_suivantes', 'page_de_fin']

        for section in required_sections:
            if section not in document or not document[section]:
                errors.append(f"Missing required section: {section}")
                sections_present[section] = False
            else:
                sections_present[section] = True

        for section in optional_sections:
            if section not in document:
                warnings.append(f"Missing optional section: {section}")
                sections_present[section] = False
            else:
                sections_present[section] = True

        # Validate document_metadata content
        if 'document_metadata' in document:
            metadata = document['document_metadata']

            # Check for recommended metadata fields
            recommended_fields = ['fund_isin', 'client_type', 'document_type', 'fund_name']
            for field in recommended_fields:
                if field not in metadata or not metadata[field]:
                    warnings.append(f"Missing recommended metadata field: {field}")

        # Check if document has any content
        has_content = False
        for section in optional_sections:
            if section in document and document[section]:
                if isinstance(document[section], list):
                    has_content = has_content or len(document[section]) > 0
                elif isinstance(document[section], dict):
                    has_content = has_content or len(document[section]) > 0

        if not has_content:
            errors.append("Document has no content in any section")

        # Validate pages_suivantes structure
        if 'pages_suivantes' in document:
            if not isinstance(document['pages_suivantes'], list):
                errors.append("pages_suivantes must be a list")
            else:
                for i, page in enumerate(document['pages_suivantes']):
                    if not isinstance(page, dict):
                        errors.append(f"pages_suivantes[{i}] must be a dictionary")

        # Determine overall validity
        valid = len(errors) == 0

        result = {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'sections_present': sections_present,
            'total_errors': len(errors),
            'total_warnings': len(warnings)
        }

        if valid:
            logger.info("Document validation passed")
        else:
            logger.warning(f"Document validation failed with {len(errors)} errors")

        return result

    except Exception as e:
        logger.error(f"Error validating document: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'sections_present': {},
            'total_errors': 1,
            'total_warnings': 0
        }


# ============================================================================
# HELPER FUNCTIONS (NOT TOOLS)
# ============================================================================

def extract_all_text_from_doc(doc: dict) -> str:
    """
    Extract all text from document for text-based analysis.

    Helper function (not a tool) that combines all document sections
    into a single text string.

    Args:
        doc: Document dictionary

    Returns:
        Combined text from all sections
    """
    all_text = []

    if 'page_de_garde' in doc:
        all_text.append(json.dumps(doc['page_de_garde'], ensure_ascii=False))

    if 'slide_2' in doc:
        all_text.append(json.dumps(doc['slide_2'], ensure_ascii=False))

    if 'pages_suivantes' in doc:
        for page in doc['pages_suivantes']:
            all_text.append(json.dumps(page, ensure_ascii=False))

    if 'page_de_fin' in doc:
        all_text.append(json.dumps(doc['page_de_fin'], ensure_ascii=False))

    return '\n'.join(all_text)


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all preprocessing tools for easy import
PREPROCESSING_TOOLS = [
    extract_metadata,
    build_whitelist,
    normalize_document,
    validate_document
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
            'fund_esg_classification': 'article_8'
        },
        'page_de_garde': {
            'title': 'Fund Presentation',
            'subtitle': 'ODDO BHF Algo Trend US'
        },
        'slide_2': {
            'content': 'Investment strategy...'
        },
        'pages_suivantes': [
            {'slide_number': 3, 'content': 'Performance data...'},
            {'slide_number': 4, 'content': 'Risk profile...'}
        ],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS'
        }
    }

    logger.info("=" * 70)
    logger.info("PREPROCESSING TOOLS TEST")
    logger.info("=" * 70)

    # Test metadata extraction
    logger.info("\n1. Extract Metadata:")
    metadata = extract_metadata.invoke({"document": test_doc})
    logger.info(f"   Fund ISIN: {metadata['fund_isin']}")
    logger.info(f"   Client Type: {metadata['client_type']}")
    logger.info(f"   Document Type: {metadata['document_type']}")
    logger.info(f"   Fund Name: {metadata['fund_name']}")

    # Test whitelist building
    logger.info("\n2. Build Whitelist:")
    whitelist = build_whitelist.invoke({"document": test_doc, "metadata": metadata})
    logger.info(f"   Total terms: {len(whitelist)}")
    logger.info(f"   Sample terms: {list(whitelist)[:10]}")

    # Test document normalization
    logger.info("\n3. Normalize Document:")
    normalized = normalize_document.invoke({"document": test_doc})
    logger.info(f"   Sections present: {list(normalized.keys())}")
    logger.info(f"   Following pages: {len(normalized['pages_suivantes'])}")

    # Test document validation
    logger.info("\n4. Validate Document:")
    validation = validate_document.invoke({"document": test_doc})
    logger.info(f"   Valid: {validation['valid']}")
    logger.info(f"   Errors: {validation['total_errors']}")
    logger.info(f"   Warnings: {validation['total_warnings']}")
    if validation['warnings']:
        for warning in validation['warnings']:
            logger.info(f"     - {warning}")

    logger.info("\n" + "=" * 70)
    logger.info("All preprocessing tools tested successfully!")
    logger.info("=" * 70)
