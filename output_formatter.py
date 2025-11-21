#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Formatter - JSON Output Format Compatibility
Ensures enhanced violations match existing structure while adding optional AI fields
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Formats compliance check results to maintain backward compatibility
    while optionally including AI-enhanced fields
    """
    
    def __init__(self, legacy_mode: bool = False, include_ai_fields: bool = True):
        """
        Initialize output formatter
        
        Args:
            legacy_mode: If True, output only legacy fields
            include_ai_fields: If True, include AI-specific fields as extensions
        """
        self.legacy_mode = legacy_mode
        self.include_ai_fields = include_ai_fields
        logger.info(f"OutputFormatter initialized (legacy_mode={legacy_mode}, include_ai_fields={include_ai_fields})")
    
    def format_violation(self, violation: Dict) -> Dict:
        """
        Format a single violation to match expected structure
        
        Args:
            violation: Violation dict from hybrid checker or legacy checker
            
        Returns:
            Formatted violation dict
        """
        # Core fields (always included)
        formatted = {
            'rule_id': self._extract_rule_id(violation),
            'rule_text': violation.get('rule', 'Unknown rule'),
            'severity': violation.get('severity', 'CRITICAL'),
            'location': violation.get('location', 'Unknown'),
            'slide_number': violation.get('slide', 'Unknown'),
            'evidence': violation.get('evidence', ''),
            'message': violation.get('message', 'Violation detected'),
            'confidence': violation.get('confidence', 0)
        }
        
        # Add AI-enhanced fields if not in legacy mode
        if not self.legacy_mode and self.include_ai_fields:
            formatted['ai_reasoning'] = violation.get('ai_reasoning', '')
            formatted['status'] = violation.get('status', '')
            formatted['needs_review'] = violation.get('needs_review', False)
            formatted['method'] = violation.get('method', 'UNKNOWN')
            
            # Add metadata about AI processing
            if 'ai_confidence' in violation:
                formatted['ai_confidence'] = violation['ai_confidence']
            if 'rule_confidence' in violation:
                formatted['rule_confidence'] = violation['rule_confidence']
        
        return formatted
    
    def format_violations_by_category(self, violations: List[Dict]) -> Dict[str, Any]:
        """
        Group violations by category
        
        Args:
            violations: List of violation dicts
            
        Returns:
            Dict with violations grouped by category
        """
        violations_by_category = {}
        
        for violation in violations:
            category = violation.get('type', 'UNKNOWN')
            
            if category not in violations_by_category:
                violations_by_category[category] = {
                    'count': 0,
                    'violations': []
                }
            
            violations_by_category[category]['count'] += 1
            violations_by_category[category]['violations'].append(
                self.format_violation(violation)
            )
        
        return violations_by_category
    
    def format_summary(self, violations: List[Dict]) -> Dict[str, int]:
        """
        Create summary statistics
        
        Args:
            violations: List of violation dicts
            
        Returns:
            Summary dict with counts by severity
        """
        summary = {
            'total_violations': len(violations),
            'critical_violations': sum(1 for v in violations if v.get('severity') == 'CRITICAL'),
            'major_violations': sum(1 for v in violations if v.get('severity') == 'MAJOR'),
            'warnings': sum(1 for v in violations if v.get('severity') == 'WARNING')
        }
        
        # Add AI-specific summary if not in legacy mode
        if not self.legacy_mode and self.include_ai_fields:
            summary['ai_detected'] = sum(1 for v in violations if 'AI_DETECTED' in v.get('status', ''))
            summary['verified_by_both'] = sum(1 for v in violations if v.get('status') == 'VERIFIED_BY_BOTH')
            summary['needs_review'] = sum(1 for v in violations if v.get('needs_review', False))
            summary['avg_confidence'] = sum(v.get('confidence', 0) for v in violations) / len(violations) if violations else 0
        
        return summary
    
    def format_document_info(self, json_file_path: str, metadata: Dict) -> Dict[str, Any]:
        """
        Format document information
        
        Args:
            json_file_path: Path to input JSON file
            metadata: Document metadata
            
        Returns:
            Formatted document info dict
        """
        doc_info = {
            'filename': json_file_path,
            'fund_name': metadata.get('fund_name', 'Unknown'),
            'fund_isin': metadata.get('fund_isin', ''),
            'client_type': metadata.get('client_type', 'retail'),
            'document_type': metadata.get('document_type', 'fund_presentation'),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add AI processing info if not in legacy mode
        if not self.legacy_mode and self.include_ai_fields:
            doc_info['processing_mode'] = 'hybrid_ai_rules'
            doc_info['ai_enabled'] = True
        else:
            doc_info['processing_mode'] = 'rules_only'
            doc_info['ai_enabled'] = False
        
        return doc_info
    
    def format_complete_output(self, json_file_path: str, metadata: Dict, 
                              violations: List[Dict], additional_data: Optional[Dict] = None) -> Dict:
        """
        Format complete output structure
        
        Args:
            json_file_path: Path to input JSON file
            metadata: Document metadata
            violations: List of violations
            additional_data: Optional additional data to include
            
        Returns:
            Complete formatted output dict
        """
        output = {
            'document_info': self.format_document_info(json_file_path, metadata),
            'summary': self.format_summary(violations),
            'violations_by_category': self.format_violations_by_category(violations),
            'all_violations': violations
        }
        
        # Add additional data if provided
        if additional_data:
            if not self.legacy_mode:
                output['additional_data'] = additional_data
        
        return output
    
    def save_output(self, output: Dict, output_path: str):
        """
        Save formatted output to JSON file
        
        Args:
            output: Formatted output dict
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"Output saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            raise
    
    def validate_output_structure(self, output: Dict) -> bool:
        """
        Validate that output matches expected structure
        
        Args:
            output: Output dict to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['document_info', 'summary', 'violations_by_category', 'all_violations']
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate document_info structure
        doc_info_fields = ['filename', 'fund_name', 'fund_isin', 'client_type', 'document_type', 'analysis_date']
        for field in doc_info_fields:
            if field not in output['document_info']:
                logger.error(f"Missing document_info field: {field}")
                return False
        
        # Validate summary structure
        summary_fields = ['total_violations', 'critical_violations', 'major_violations', 'warnings']
        for field in summary_fields:
            if field not in output['summary']:
                logger.error(f"Missing summary field: {field}")
                return False
        
        # Validate violations structure
        if not isinstance(output['violations_by_category'], dict):
            logger.error("violations_by_category must be a dict")
            return False
        
        if not isinstance(output['all_violations'], list):
            logger.error("all_violations must be a list")
            return False
        
        logger.info("Output structure validation passed")
        return True
    
    def _extract_rule_id(self, violation: Dict) -> str:
        """Extract rule ID from violation"""
        rule = violation.get('rule', '')
        
        # Try to extract ID from rule text (e.g., "STRUCT_001: Description")
        if ':' in rule:
            return rule.split(':')[0].strip()
        
        # Generate ID from rule text
        if len(rule) > 20:
            return rule[:20].replace(' ', '_').upper()
        
        return rule.replace(' ', '_').upper() if rule else 'UNKNOWN'
    
    def convert_legacy_to_enhanced(self, legacy_output: Dict) -> Dict:
        """
        Convert legacy output format to enhanced format
        
        Args:
            legacy_output: Output in legacy format
            
        Returns:
            Enhanced output format
        """
        # If already in enhanced format, return as-is
        if 'document_info' in legacy_output and 'summary' in legacy_output:
            return legacy_output
        
        # Convert from legacy format
        enhanced = {
            'document_info': {
                'filename': legacy_output.get('filename', 'Unknown'),
                'fund_name': legacy_output.get('fund_name', 'Unknown'),
                'fund_isin': legacy_output.get('fund_isin', ''),
                'client_type': legacy_output.get('client_type', 'retail'),
                'document_type': legacy_output.get('document_type', 'fund_presentation'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'processing_mode': 'rules_only',
                'ai_enabled': False
            },
            'summary': {
                'total_violations': legacy_output.get('total_violations', 0),
                'critical_violations': legacy_output.get('critical_violations', 0),
                'major_violations': legacy_output.get('major_violations', 0),
                'warnings': legacy_output.get('warnings', 0)
            },
            'violations_by_category': legacy_output.get('violations_by_category', {}),
            'all_violations': legacy_output.get('violations', [])
        }
        
        return enhanced


def create_formatter(config: Optional[Dict] = None) -> OutputFormatter:
    """
    Create output formatter from configuration
    
    Args:
        config: Configuration dict
        
    Returns:
        OutputFormatter instance
    """
    if config is None:
        config = {}
    
    legacy_mode = config.get('use_legacy_format', False)
    include_ai_fields = not legacy_mode and config.get('ai_enabled', True)
    
    return OutputFormatter(legacy_mode=legacy_mode, include_ai_fields=include_ai_fields)


if __name__ == "__main__":
    # Test output formatter
    print("="*70)
    print("Output Formatter - JSON Compatibility Test")
    print("="*70)
    
    # Test with sample violations
    sample_violations = [
        {
            'type': 'STRUCTURE',
            'severity': 'CRITICAL',
            'slide': 'Cover Page',
            'location': 'Header',
            'rule': 'STRUCT_001: Promotional mention required',
            'message': 'Missing promotional document mention',
            'evidence': 'No promotional indication found',
            'confidence': 95,
            'ai_reasoning': 'AI detected missing promotional language',
            'status': 'VERIFIED_BY_BOTH',
            'needs_review': False,
            'ai_confidence': 90,
            'rule_confidence': 85
        },
        {
            'type': 'PERFORMANCE',
            'severity': 'MAJOR',
            'slide': 'Page 5',
            'location': 'Performance section',
            'rule': 'PERF_002: Performance disclaimer required',
            'message': 'Missing performance disclaimer',
            'evidence': 'Performance data without disclaimer',
            'confidence': 75,
            'ai_reasoning': 'AI detected performance claims without disclaimer',
            'status': 'AI_DETECTED_VARIATION',
            'needs_review': True,
            'ai_confidence': 75,
            'rule_confidence': 60
        }
    ]
    
    sample_metadata = {
        'fund_name': 'Test Fund',
        'fund_isin': 'FR0000000000',
        'client_type': 'retail',
        'document_type': 'fund_presentation'
    }
    
    # Test enhanced format
    print("\n📊 Test 1: Enhanced Format (with AI fields)")
    formatter = OutputFormatter(legacy_mode=False, include_ai_fields=True)
    output = formatter.format_complete_output('test.json', sample_metadata, sample_violations)
    
    print(f"  Document Info: {output['document_info']['processing_mode']}")
    print(f"  Total Violations: {output['summary']['total_violations']}")
    print(f"  AI Detected: {output['summary'].get('ai_detected', 0)}")
    print(f"  Needs Review: {output['summary'].get('needs_review', 0)}")
    print(f"  Validation: {'✓ PASS' if formatter.validate_output_structure(output) else '✗ FAIL'}")
    
    # Test legacy format
    print("\n📊 Test 2: Legacy Format (no AI fields)")
    formatter_legacy = OutputFormatter(legacy_mode=True, include_ai_fields=False)
    output_legacy = formatter_legacy.format_complete_output('test.json', sample_metadata, sample_violations)
    
    print(f"  Document Info: {output_legacy['document_info']['processing_mode']}")
    print(f"  Total Violations: {output_legacy['summary']['total_violations']}")
    print(f"  Has AI fields: {'ai_detected' in output_legacy['summary']}")
    print(f"  Validation: {'✓ PASS' if formatter_legacy.validate_output_structure(output_legacy) else '✗ FAIL'}")
    
    # Test violation formatting
    print("\n📊 Test 3: Violation Formatting")
    formatted_violation = formatter.format_violation(sample_violations[0])
    print(f"  Rule ID: {formatted_violation['rule_id']}")
    print(f"  Confidence: {formatted_violation['confidence']}%")
    print(f"  Has AI reasoning: {'ai_reasoning' in formatted_violation}")
    print(f"  Has status: {'status' in formatted_violation}")
    
    formatted_violation_legacy = formatter_legacy.format_violation(sample_violations[0])
    print(f"\n  Legacy format:")
    print(f"  Rule ID: {formatted_violation_legacy['rule_id']}")
    print(f"  Confidence: {formatted_violation_legacy['confidence']}%")
    print(f"  Has AI reasoning: {'ai_reasoning' in formatted_violation_legacy}")
    print(f"  Has status: {'status' in formatted_violation_legacy}")
    
    print("\n" + "="*70)
    print("✓ Output formatter tests complete")
    print("="*70)
