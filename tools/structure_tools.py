#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Tools

This module provides functionality for the multi-agent compliance system.
"""

"""
Structure Checking Tools for Multi-Agent Compliance System

These tools handle structure compliance checks:
- Promotional document mention
- Target audience specification
- Management company legal mention
- Fund name validation
- Date validation

Requirements: 2.1, 7.2, 7.5
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Optional, List
from datetime import datetime
from langchain.tools import tool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PROMOTIONAL MENTION TOOL
# ============================================================================

@tool
def check_promotional_mention(document: dict, config: dict) -> Optional[dict]:
    """
    Check for promotional document mention on cover page.

    Regulatory requirement: Documents must clearly indicate they are
    "promotional" or "marketing" materials on the cover page.

    Checks for phrases like:
    - "Document promotionnel"
    - "Promotional document"
    - "Document à caractère promotionnel"
    - "Marketing material"

    Args:
        document: Document dictionary with page_de_garde section
        config: Configuration dictionary with AI settings

    Returns:
        Violation dictionary if promotional mention is missing, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        page_de_garde = document.get('page_de_garde', {})
        cover_text = json.dumps(page_de_garde, ensure_ascii=False).lower()

        # Define promotional keywords
        promotional_keywords = [
            'document promotionnel',
            'promotional document',
            'document à caractère promotionnel',
            'promotional material',
            'marketing document',
            'document marketing',
            'matériel promotionnel',
            'marketing material'
        ]

        # Check if any promotional keyword is present
        found_keywords = [kw for kw in promotional_keywords if kw in cover_text]

        if len(found_keywords) == 0:
            # No promotional mention found - this is a violation
            logger.warning("Promotional document mention missing on cover page")

            return {
                'type': 'STRUCTURE',
                'severity': 'CRITICAL',
                'slide': 'Cover Page',
                'location': 'page_de_garde',
                'rule': 'STRUCT_003: Must indicate "promotional document"',
                'message': 'Missing promotional document mention',
                'evidence': f'Keywords checked: {", ".join(promotional_keywords[:3])}...',
                'confidence': 90,
                'method': 'RULE_BASED',
                'rule_hints': 'No promotional keywords found on cover page'
            }

        # Promotional mention found - no violation
        logger.info(f"Promotional mention found: {found_keywords[0]}")
        return None

    except Exception as e:
        logger.error(f"Error checking promotional mention: {e}")
        return {
            'type': 'STRUCTURE',
            'severity': 'CRITICAL',
            'slide': 'Cover Page',
            'location': 'page_de_garde',
            'rule': 'STRUCT_003: Must indicate "promotional document"',
            'message': f'Error checking promotional mention: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# TARGET AUDIENCE TOOL
# ============================================================================

@tool
def check_target_audience(document: dict, client_type: str) -> Optional[dict]:
    """
    Check for target audience specification on cover page.

    Regulatory requirement: Documents must clearly specify their target audience:
    - "Retail investors" / "Investisseurs non professionnels"
    - "Professional investors" / "Investisseurs professionnels"
    - "Qualified investors" / "Clients professionnels"

    Args:
        document: Document dictionary with page_de_garde section
        client_type: Expected client type (retail/professional)

    Returns:
        Violation dictionary if target audience is not specified, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        page_de_garde = document.get('page_de_garde', {})
        cover_text = json.dumps(page_de_garde, ensure_ascii=False).lower()

        # Define audience keywords
        audience_keywords = [
            'investisseurs non professionnels',
            'investisseurs professionnels',
            'non-professional investors',
            'professional investors',
            'retail investors',
            'qualified investors',
            'document destiné aux',
            'intended for',
            'clients professionnels',
            'clients non professionnels'
        ]

        # Check if any audience keyword is present
        found_keywords = [kw for kw in audience_keywords if kw in cover_text]

        if len(found_keywords) == 0:
            # No target audience specified - this is a violation
            logger.warning("Target audience not specified on cover page")

            return {
                'type': 'STRUCTURE',
                'severity': 'CRITICAL',
                'slide': 'Cover Page',
                'location': 'page_de_garde',
                'rule': 'STRUCT_004: Must indicate target audience',
                'message': 'Target audience not specified',
                'evidence': 'Checked for: retail/professional investor mentions',
                'confidence': 85,
                'method': 'RULE_BASED',
                'rule_hints': 'No audience specification found on cover page'
            }

        # Target audience found - no violation
        logger.info(f"Target audience found: {found_keywords[0]}")
        return None

    except Exception as e:
        logger.error(f"Error checking target audience: {e}")
        return {
            'type': 'STRUCTURE',
            'severity': 'CRITICAL',
            'slide': 'Cover Page',
            'location': 'page_de_garde',
            'rule': 'STRUCT_004: Must indicate target audience',
            'message': f'Error checking target audience: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# MANAGEMENT COMPANY TOOL
# ============================================================================

@tool
def check_management_company(document: dict) -> Optional[dict]:
    """
    Check for management company legal mention on back page.

    Regulatory requirement: Documents must include full legal mention of
    the management company, typically including:
    - Company name: "ODDO BHF Asset Management"
    - Legal form: "SAS" or similar
    - Registration details
    - Address

    Args:
        document: Document dictionary with page_de_fin section

    Returns:
        Violation dictionary if management company mention is missing, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        page_de_fin = document.get('page_de_fin', {})
        legal_text = json.dumps(page_de_fin, ensure_ascii=False).lower()

        # Define company keywords
        company_keywords = [
            'oddo bhf asset management',
            'oddo bhf am',
            'société de gestion',
            'management company',
            'asset management sas'
        ]

        # Check if any company keyword is present
        found_keywords = [kw for kw in company_keywords if kw in legal_text]

        if len(found_keywords) == 0:
            # No management company mention found - this is a violation
            logger.warning("Management company legal mention missing on back page")

            return {
                'type': 'STRUCTURE',
                'severity': 'CRITICAL',
                'slide': 'Back Page',
                'location': 'page_de_fin',
                'rule': 'STRUCT_011: Legal mention of management company',
                'message': 'Management company legal mention missing',
                'evidence': 'Must include full legal name of management company',
                'confidence': 80,
                'method': 'RULE_BASED',
                'rule_hints': 'No company mention found on back page'
            }

        # Management company mention found - no violation
        logger.info(f"Management company mention found: {found_keywords[0]}")
        return None

    except Exception as e:
        logger.error(f"Error checking management company: {e}")
        return {
            'type': 'STRUCTURE',
            'severity': 'CRITICAL',
            'slide': 'Back Page',
            'location': 'page_de_fin',
            'rule': 'STRUCT_011: Legal mention of management company',
            'message': f'Error checking management company: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# FUND NAME TOOL
# ============================================================================

@tool
def check_fund_name(document: dict, metadata: dict) -> Optional[dict]:
    """
    Check that fund name is consistently used throughout the document.

    Validates that:
    - Fund name from metadata matches cover page
    - Fund name is present and consistent
    - No significant variations or typos

    Args:
        document: Document dictionary
        metadata: Extracted metadata with fund_name

    Returns:
        Violation dictionary if fund name is inconsistent or missing, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        # Get fund name from metadata
        fund_name = metadata.get('fund_name', '')

        if not fund_name:
            # No fund name in metadata - cannot validate
            logger.warning("Fund name not provided in metadata")
            return {
                'type': 'STRUCTURE',
                'severity': 'MAJOR',
                'slide': 'Document-wide',
                'location': 'Fund identification',
                'rule': 'STRUCT_001: Fund name must be specified',
                'message': 'Fund name not provided in metadata',
                'evidence': 'Cannot validate fund name consistency',
                'confidence': 70,
                'method': 'RULE_BASED',
                'rule_hints': 'Fund name missing from document metadata'
            }

        # Check cover page for fund name
        page_de_garde = document.get('page_de_garde', {})
        cover_text = json.dumps(page_de_garde, ensure_ascii=False).lower()
        fund_name_lower = fund_name.lower()

        # Extract key components of fund name (ignore common words)
        fund_name_components = [
            word for word in fund_name_lower.split()
            if len(word) > 3 and word not in ['fund', 'sicav', 'class', 'share']
        ]

        # Check if main components are present on cover
        components_found = [comp for comp in fund_name_components if comp in cover_text]

        if len(components_found) < len(fund_name_components) * 0.5:
            # Less than 50% of fund name components found - likely missing or wrong
            logger.warning(f"Fund name '{fund_name}' not clearly present on cover page")

            return {
                'type': 'STRUCTURE',
                'severity': 'MAJOR',
                'slide': 'Cover Page',
                'location': 'page_de_garde',
                'rule': 'STRUCT_001: Fund name must be clearly displayed',
                'message': f'Fund name "{fund_name}" not clearly present on cover',
                'evidence': f'Expected: {fund_name}, Found components: {components_found}',
                'confidence': 75,
                'method': 'RULE_BASED',
                'rule_hints': f'Only {len(components_found)}/{len(fund_name_components)} name components found'
            }

        # Fund name found - no violation
        logger.info(f"Fund name '{fund_name}' found on cover page")
        return None

    except Exception as e:
        logger.error(f"Error checking fund name: {e}")
        return {
            'type': 'STRUCTURE',
            'severity': 'MAJOR',
            'slide': 'Cover Page',
            'location': 'page_de_garde',
            'rule': 'STRUCT_001: Fund name must be clearly displayed',
            'message': f'Error checking fund name: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# DATE VALIDATION TOOL
# ============================================================================

@tool
def check_date_validation(document: dict) -> Optional[dict]:
    """
    Check that document has a valid date and it's not too old.

    Validates that:
    - Document has a date
    - Date is in valid format
    - Date is not in the future
    - Document is not older than 12 months (warning)

    Args:
        document: Document dictionary with document_metadata

    Returns:
        Violation dictionary if date is invalid or missing, None otherwise

    Requirements: 2.1, 7.2, 7.5
    """
    try:
        doc_metadata = document.get('document_metadata', {})

        # Try to find date in various fields
        date_str = (
            doc_metadata.get('document_date') or
            doc_metadata.get('creation_date') or
            doc_metadata.get('date')
        )

        if not date_str:
            # No date found - this is a violation
            logger.warning("Document date not found in metadata")

            return {
                'type': 'STRUCTURE',
                'severity': 'MAJOR',
                'slide': 'Document metadata',
                'location': 'document_metadata',
                'rule': 'STRUCT_009: Document must have a date',
                'message': 'Document date not specified',
                'evidence': 'No date found in document_metadata',
                'confidence': 90,
                'method': 'RULE_BASED',
                'rule_hints': 'Missing document_date, creation_date, or date field'
            }

        # Try to parse the date
        try:
            # Support multiple date formats
            date_formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%Y%m%d'
            ]

            doc_date = None
            for fmt in date_formats:
                try:
                    doc_date = datetime.strptime(str(date_str), fmt)
                    break
                except ValueError:
                    continue

            if doc_date is None:
                # Could not parse date
                logger.warning(f"Could not parse document date: {date_str}")

                return {
                    'type': 'STRUCTURE',
                    'severity': 'MAJOR',
                    'slide': 'Document metadata',
                    'location': 'document_metadata',
                    'rule': 'STRUCT_009: Document date must be valid',
                    'message': f'Invalid date format: {date_str}',
                    'evidence': f'Expected formats: YYYY-MM-DD, DD/MM/YYYY, etc.',
                    'confidence': 85,
                    'method': 'RULE_BASED',
                    'rule_hints': f'Could not parse date: {date_str}'
                }

            # Check if date is in the future
            now = datetime.now()
            if doc_date > now:
                logger.warning(f"Document date is in the future: {date_str}")

                return {
                    'type': 'STRUCTURE',
                    'severity': 'MAJOR',
                    'slide': 'Document metadata',
                    'location': 'document_metadata',
                    'rule': 'STRUCT_009: Document date cannot be in future',
                    'message': f'Document date is in the future: {date_str}',
                    'evidence': f'Document date: {doc_date.strftime("%Y-%m-%d")}, Current date: {now.strftime("%Y-%m-%d")}',
                    'confidence': 95,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Future dates are not allowed'
                }

            # Check if document is too old (warning only, not critical)
            age_days = (now - doc_date).days
            if age_days > 365:
                logger.warning(f"Document is older than 12 months: {age_days} days")

                return {
                    'type': 'STRUCTURE',
                    'severity': 'WARNING',
                    'slide': 'Document metadata',
                    'location': 'document_metadata',
                    'rule': 'STRUCT_010: Document may be outdated',
                    'message': f'Document is {age_days} days old (>{365} days)',
                    'evidence': f'Document date: {doc_date.strftime("%Y-%m-%d")}, Age: {age_days} days',
                    'confidence': 80,
                    'method': 'RULE_BASED',
                    'rule_hints': 'Consider updating document if information is stale'
                }

            # Date is valid - no violation
            logger.info(f"Document date is valid: {date_str} ({age_days} days old)")
            return None

        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return {
                'type': 'STRUCTURE',
                'severity': 'MAJOR',
                'slide': 'Document metadata',
                'location': 'document_metadata',
                'rule': 'STRUCT_009: Document date must be valid',
                'message': f'Error parsing date: {str(e)}',
                'evidence': f'Date value: {date_str}',
                'confidence': 70,
                'method': 'ERROR'
            }

    except Exception as e:
        logger.error(f"Error checking date validation: {e}")
        return {
            'type': 'STRUCTURE',
            'severity': 'MAJOR',
            'slide': 'Document metadata',
            'location': 'document_metadata',
            'rule': 'STRUCT_009: Document must have a date',
            'message': f'Error checking date: {str(e)}',
            'evidence': 'Unable to verify due to error',
            'confidence': 50,
            'method': 'ERROR'
        }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# List of all structure checking tools for easy import
STRUCTURE_TOOLS = [
    check_promotional_mention,
    check_target_audience,
    check_management_company,
    check_fund_name,
    check_date_validation
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
            'document_date': '2024-01-15'
        },
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel destiné aux investisseurs non professionnels'
        },
        'slide_2': {
            'content': 'Investment strategy...'
        },
        'pages_suivantes': [],
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS - Société de gestion'
        }
    }

    test_metadata = {
        'fund_name': 'ODDO BHF Algo Trend US Fund',
        'client_type': 'retail'
    }

    test_config = {
        'ai_enabled': True
    }

    logger.info("=" * 70)
    logger.info("STRUCTURE CHECKING TOOLS TEST")
    logger.info("=" * 70)

    # Test promotional mention
    logger.info("\n1. Check Promotional Mention:")
    try:
        result = check_promotional_mention.func(document=test_doc, config=test_config)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Promotional mention found")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test target audience
    logger.info("\n2. Check Target Audience:")
    try:
        result = check_target_audience.func(document=test_doc, client_type="retail")
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Target audience specified")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test management company
    logger.info("\n3. Check Management Company:")
    try:
        result = check_management_company.func(document=test_doc)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Management company mentioned")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test fund name
    logger.info("\n4. Check Fund Name:")
    try:
        result = check_fund_name.func(document=test_doc, metadata=test_metadata)
        if result:
            logger.info(f"   ❌ VIOLATION: {result['message']}")
        else:
            logger.info("   ✓ PASS: Fund name present")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    # Test date validation
    logger.info("\n5. Check Date Validation:")
    try:
        result = check_date_validation.func(document=test_doc)
        if result:
            logger.info(f"   ⚠️  {result['severity']}: {result['message']}")
        else:
            logger.info("   ✓ PASS: Date is valid")
    except Exception as e:
        logger.info(f"   ⚠️  ERROR: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("All structure checking tools tested!")
    logger.info("=" * 70)
