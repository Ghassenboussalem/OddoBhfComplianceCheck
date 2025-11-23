#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for structure checking tools

Tests:
- check_promotional_mention
- check_target_audience
- check_management_company
- check_fund_name
- check_date_validation
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.structure_tools import (
    check_promotional_mention,
    check_target_audience,
    check_management_company,
    check_fund_name,
    check_date_validation
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Test configuration"""
    return {'ai_enabled': True}


@pytest.fixture
def valid_cover_page():
    """Valid cover page with all required elements"""
    return {
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund',
            'subtitle': 'Document promotionnel destiné aux investisseurs non professionnels'
        }
    }


@pytest.fixture
def valid_back_page():
    """Valid back page with management company"""
    return {
        'page_de_fin': {
            'legal': 'ODDO BHF Asset Management SAS - Société de gestion'
        }
    }


# ============================================================================
# PROMOTIONAL MENTION TESTS
# ============================================================================

def test_promotional_mention_present(valid_cover_page, config):
    """Test when promotional mention is present"""
    result = check_promotional_mention.func(document=valid_cover_page, config=config)
    assert result is None


def test_promotional_mention_missing(config):
    """Test when promotional mention is missing"""
    doc = {'page_de_garde': {'title': 'Fund Presentation'}}
    result = check_promotional_mention.func(document=doc, config=config)
    
    assert result is not None
    assert result['type'] == 'STRUCTURE'
    assert result['severity'] == 'CRITICAL'
    assert 'promotional' in result['message'].lower()


def test_promotional_mention_english(config):
    """Test English promotional mention"""
    doc = {'page_de_garde': {'text': 'Promotional document for investors'}}
    result = check_promotional_mention.func(document=doc, config=config)
    assert result is None


# ============================================================================
# TARGET AUDIENCE TESTS
# ============================================================================

def test_target_audience_present(valid_cover_page):
    """Test when target audience is specified"""
    result = check_target_audience.func(document=valid_cover_page, client_type='retail')
    assert result is None


def test_target_audience_missing():
    """Test when target audience is missing"""
    doc = {'page_de_garde': {'title': 'Fund Presentation'}}
    result = check_target_audience.func(document=doc, client_type='retail')
    
    assert result is not None
    assert result['type'] == 'STRUCTURE'
    assert 'audience' in result['message'].lower()


def test_target_audience_professional():
    """Test professional investor mention"""
    doc = {'page_de_garde': {'text': 'Document for professional investors'}}
    result = check_target_audience.func(document=doc, client_type='professional')
    assert result is None


# ============================================================================
# MANAGEMENT COMPANY TESTS
# ============================================================================

def test_management_company_present(valid_back_page):
    """Test when management company is mentioned"""
    result = check_management_company.func(document=valid_back_page)
    assert result is None


def test_management_company_missing():
    """Test when management company is missing"""
    doc = {'page_de_fin': {'text': 'Legal information'}}
    result = check_management_company.func(document=doc)
    
    assert result is not None
    assert result['type'] == 'STRUCTURE'
    assert 'management company' in result['message'].lower()


def test_management_company_abbreviated():
    """Test abbreviated company name"""
    doc = {'page_de_fin': {'legal': 'ODDO BHF AM SAS'}}
    result = check_management_company.func(document=doc)
    assert result is None


# ============================================================================
# FUND NAME TESTS
# ============================================================================

def test_fund_name_present():
    """Test when fund name is present on cover"""
    doc = {
        'page_de_garde': {
            'title': 'ODDO BHF Algo Trend US Fund'
        }
    }
    metadata = {'fund_name': 'ODDO BHF Algo Trend US Fund'}
    result = check_fund_name.func(document=doc, metadata=metadata)
    assert result is None


def test_fund_name_missing():
    """Test when fund name is missing from cover"""
    doc = {'page_de_garde': {'title': 'Investment Presentation'}}
    metadata = {'fund_name': 'ODDO BHF Algo Trend US Fund'}
    result = check_fund_name.func(document=doc, metadata=metadata)
    
    assert result is not None
    assert result['type'] == 'STRUCTURE'


def test_fund_name_no_metadata():
    """Test when fund name not in metadata"""
    doc = {'page_de_garde': {'title': 'Fund Presentation'}}
    metadata = {}
    result = check_fund_name.func(document=doc, metadata=metadata)
    
    assert result is not None
    assert 'not provided' in result['message'].lower()


def test_fund_name_partial_match():
    """Test partial fund name match"""
    doc = {'page_de_garde': {'title': 'ODDO BHF Algo Trend'}}
    metadata = {'fund_name': 'ODDO BHF Algo Trend US Fund'}
    result = check_fund_name.func(document=doc, metadata=metadata)
    # Should pass with partial match
    assert result is None


# ============================================================================
# DATE VALIDATION TESTS
# ============================================================================

def test_date_validation_valid():
    """Test with valid recent date"""
    today = datetime.now()
    date_str = today.strftime('%Y-%m-%d')
    doc = {'document_metadata': {'document_date': date_str}}
    result = check_date_validation.func(document=doc)
    assert result is None


def test_date_validation_missing():
    """Test when date is missing"""
    doc = {'document_metadata': {}}
    result = check_date_validation.func(document=doc)
    
    assert result is not None
    assert result['type'] == 'STRUCTURE'
    assert 'date' in result['message'].lower()


def test_date_validation_future():
    """Test with future date"""
    future = datetime.now() + timedelta(days=30)
    date_str = future.strftime('%Y-%m-%d')
    doc = {'document_metadata': {'document_date': date_str}}
    result = check_date_validation.func(document=doc)
    
    assert result is not None
    assert 'future' in result['message'].lower()


def test_date_validation_old():
    """Test with old date (>12 months)"""
    old = datetime.now() - timedelta(days=400)
    date_str = old.strftime('%Y-%m-%d')
    doc = {'document_metadata': {'document_date': date_str}}
    result = check_date_validation.func(document=doc)
    
    assert result is not None
    assert result['severity'] == 'WARNING'


def test_date_validation_formats():
    """Test various date formats"""
    formats = [
        '2024-01-15',
        '15/01/2024',
        '01/15/2024',
        '2024/01/15'
    ]
    
    for date_str in formats:
        doc = {'document_metadata': {'document_date': date_str}}
        result = check_date_validation.func(document=doc)
        # Should parse successfully (may warn if old)
        assert result is None or result['severity'] == 'WARNING'


def test_date_validation_invalid_format():
    """Test with invalid date format"""
    doc = {'document_metadata': {'document_date': 'invalid-date'}}
    result = check_date_validation.func(document=doc)
    
    assert result is not None
    assert 'invalid' in result['message'].lower() or 'parse' in result['message'].lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
