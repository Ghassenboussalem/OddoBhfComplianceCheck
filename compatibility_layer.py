#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward Compatibility Layer for Multi-Agent System

This module provides backward compatibility between the original check.py
and the new multi-agent check_multiagent.py system. It ensures that:

1. All existing command-line flags are supported
2. JSON output format remains compatible
3. Existing scripts and workflows continue to work
4. Optional agent metadata can be added to output

Features:
- Command-line argument translation
- Output format compatibility
- Configuration mapping
- Feature flag support
"""

import sys
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class CompatibilityLayer:
    """
    Provides backward compatibility between old and new systems
    """
    
    def __init__(self):
        """Initialize compatibility layer"""
        self.legacy_mode = False
        self.include_agent_metadata = False
        
    def translate_command_line_args(self, args: List[str]) -> Dict[str, Any]:
        """
        Translate command-line arguments from check.py format to check_multiagent.py format
        
        Supported flags from check.py:
        - --hybrid-mode=on|off
        - --rules-only
        - --context-aware=on|off
        - --ai-confidence=N
        - --review-mode
        - --review-threshold=N
        - --show-metrics
        
        Args:
            args: Command-line arguments (sys.argv)
            
        Returns:
            Dictionary of parsed options compatible with both systems
        """
        options = {
            'json_file': None,
            'hybrid_mode': None,
            'rules_only': False,
            'context_aware': None,
            'ai_confidence': None,
            'review_mode': False,
            'review_threshold': None,
            'show_metrics': False,
            'use_multiagent': False,  # Flag to indicate multi-agent mode
            'legacy_output': False    # Flag to force legacy output format
        }
        
        if len(args) < 2:
            return options
        
        # First argument is the JSON file
        options['json_file'] = args[1]
        
        # Parse additional options
        for arg in args[2:]:
            if arg.startswith('--hybrid-mode='):
                mode = arg.split('=')[1].lower()
                options['hybrid_mode'] = (mode == 'on')
                logger.info(f"Hybrid mode: {options['hybrid_mode']}")
                
            elif arg == '--rules-only':
                options['rules_only'] = True
                options['hybrid_mode'] = False
                logger.info("Rules-only mode enabled")
                
            elif arg.startswith('--context-aware='):
                mode = arg.split('=')[1].lower()
                options['context_aware'] = (mode == 'on')
                logger.info(f"Context-aware mode: {options['context_aware']}")
                
            elif arg.startswith('--ai-confidence='):
                try:
                    options['ai_confidence'] = int(arg.split('=')[1])
                    logger.info(f"AI confidence threshold: {options['ai_confidence']}")
                except ValueError:
                    logger.warning(f"Invalid AI confidence value: {arg}")
                    
            elif arg == '--review-mode':
                options['review_mode'] = True
                logger.info("Review mode enabled")
                
            elif arg.startswith('--review-threshold='):
                try:
                    options['review_threshold'] = int(arg.split('=')[1])
                    logger.info(f"Review threshold: {options['review_threshold']}")
                except ValueError:
                    logger.warning(f"Invalid review threshold value: {arg}")
                    
            elif arg == '--show-metrics':
                options['show_metrics'] = True
                logger.info("Show metrics enabled")
                
            elif arg == '--use-multiagent':
                options['use_multiagent'] = True
                logger.info("Multi-agent mode explicitly enabled")
                
            elif arg == '--legacy-output':
                options['legacy_output'] = True
                logger.info("Legacy output format enabled")
                
            else:
                logger.warning(f"Unknown argument: {arg}")
        
        return options
    
    def map_config_options(self, options: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map command-line options to configuration dictionary
        
        Args:
            options: Parsed command-line options
            base_config: Base configuration from hybrid_config.json
            
        Returns:
            Updated configuration dictionary
        """
        config = base_config.copy()
        
        # Apply hybrid mode setting
        if options['hybrid_mode'] is not None:
            config['ai_enabled'] = options['hybrid_mode']
            
        # Apply rules-only mode
        if options['rules_only']:
            config['ai_enabled'] = False
            
        # Apply context-aware setting
        if options['context_aware'] is not None:
            if 'features' not in config:
                config['features'] = {}
            config['features']['enable_context_aware_ai'] = options['context_aware']
            
        # Apply AI confidence threshold
        if options['ai_confidence'] is not None:
            if 'confidence' not in config:
                config['confidence'] = {}
            config['confidence']['threshold'] = options['ai_confidence']
            
        # Apply review threshold
        if options['review_threshold'] is not None:
            if 'hitl' not in config:
                config['hitl'] = {}
            config['hitl']['review_threshold'] = options['review_threshold']
            
        # Apply legacy output format
        if options['legacy_output']:
            config['use_legacy_format'] = True
            
        return config
    
    def ensure_output_compatibility(
        self,
        output: Dict[str, Any],
        include_agent_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Ensure output format is compatible with existing tools
        
        The output must maintain the same structure as check.py:
        {
            "document_id": "...",
            "metadata": {...},
            "violations": [...],
            "total_violations": N,
            "check_timestamp": "...",
            "ai_enabled": bool,
            "confidence_threshold": N
        }
        
        Optionally add agent metadata if requested:
        {
            ...existing fields...,
            "multi_agent": {
                "enabled": true,
                "workflow_status": "...",
                "agent_timings": {...},
                "thread_id": "...",
                "total_execution_time": N
            }
        }
        
        Args:
            output: Output dictionary from multi-agent system
            include_agent_metadata: Whether to include agent-specific metadata
            
        Returns:
            Compatible output dictionary
        """
        # Ensure all required fields exist
        compatible_output = {
            'document_id': output.get('document_id', ''),
            'metadata': output.get('metadata', {}),
            'violations': output.get('violations', []),
            'total_violations': output.get('total_violations', len(output.get('violations', []))),
            'check_timestamp': output.get('check_timestamp', ''),
        }
        
        # Add AI-related fields if present
        if 'ai_enabled' in output:
            compatible_output['ai_enabled'] = output['ai_enabled']
        if 'confidence_threshold' in output:
            compatible_output['confidence_threshold'] = output['confidence_threshold']
            
        # Add optional agent metadata
        if include_agent_metadata and 'multi_agent' in output:
            compatible_output['multi_agent'] = output['multi_agent']
            
        return compatible_output
    
    def validate_violation_format(self, violation: Dict[str, Any]) -> bool:
        """
        Validate that a violation has the required fields for compatibility
        
        Required fields:
        - type
        - severity
        - slide
        - location
        - rule
        - message
        - evidence
        
        Optional fields:
        - confidence
        - ai_reasoning
        - method
        - detected_by (agent name)
        
        Args:
            violation: Violation dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['type', 'severity', 'slide', 'location', 'rule', 'message', 'evidence']
        
        for field in required_fields:
            if field not in violation:
                logger.warning(f"Violation missing required field: {field}")
                return False
                
        return True
    
    def normalize_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize violations to ensure compatibility
        
        Args:
            violations: List of violation dictionaries
            
        Returns:
            Normalized list of violations
        """
        normalized = []
        
        for v in violations:
            # Validate format
            if not self.validate_violation_format(v):
                logger.warning(f"Skipping invalid violation: {v.get('rule', 'unknown')}")
                continue
                
            # Create normalized violation
            normalized_v = {
                'type': v['type'],
                'severity': v['severity'],
                'slide': v['slide'],
                'location': v['location'],
                'rule': v['rule'],
                'message': v['message'],
                'evidence': v['evidence']
            }
            
            # Add optional fields if present
            if 'confidence' in v:
                normalized_v['confidence'] = v['confidence']
            if 'ai_reasoning' in v:
                normalized_v['ai_reasoning'] = v['ai_reasoning']
            if 'method' in v:
                normalized_v['method'] = v['method']
                
            # Add agent metadata if not in legacy mode
            if not self.legacy_mode and 'detected_by' in v:
                normalized_v['detected_by'] = v['detected_by']
                
            normalized.append(normalized_v)
            
        return normalized
    
    def compare_outputs(
        self,
        old_output: Dict[str, Any],
        new_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare outputs from old and new systems for validation
        
        Args:
            old_output: Output from check.py
            new_output: Output from check_multiagent.py
            
        Returns:
            Comparison report
        """
        report = {
            'violations_match': False,
            'count_match': False,
            'differences': [],
            'old_count': len(old_output.get('violations', [])),
            'new_count': len(new_output.get('violations', []))
        }
        
        # Compare violation counts
        report['count_match'] = (report['old_count'] == report['new_count'])
        
        # Compare violations
        old_violations = old_output.get('violations', [])
        new_violations = new_output.get('violations', [])
        
        # Create sets of violation signatures for comparison
        old_sigs = set()
        for v in old_violations:
            sig = f"{v.get('type')}|{v.get('rule')}|{v.get('slide')}|{v.get('location')}"
            old_sigs.add(sig)
            
        new_sigs = set()
        for v in new_violations:
            sig = f"{v.get('type')}|{v.get('rule')}|{v.get('slide')}|{v.get('location')}"
            new_sigs.add(sig)
            
        # Find differences
        only_in_old = old_sigs - new_sigs
        only_in_new = new_sigs - old_sigs
        
        if only_in_old:
            report['differences'].append({
                'type': 'missing_in_new',
                'count': len(only_in_old),
                'violations': list(only_in_old)
            })
            
        if only_in_new:
            report['differences'].append({
                'type': 'new_violations',
                'count': len(only_in_new),
                'violations': list(only_in_new)
            })
            
        report['violations_match'] = (len(only_in_old) == 0 and len(only_in_new) == 0)
        
        return report
    
    def print_usage(self, program_name: str = "check_multiagent.py"):
        """
        Print usage information compatible with check.py
        
        Args:
            program_name: Name of the program
        """
        print("\n" + "="*70)
        print("COMPLIANCE CHECKER - MULTI-AGENT VERSION")
        print("="*70)
        print("\nUsage:")
        print(f"  python {program_name} <json_file> [options]")
        print("\nOptions:")
        print("  --hybrid-mode=on|off    Enable/disable AI+Rules hybrid mode")
        print("  --rules-only            Use only rule-based checking (no AI)")
        print("  --context-aware=on|off  Enable/disable AI context-aware mode")
        print("  --ai-confidence=N       Set AI confidence threshold (default: 70)")
        print("  --review-mode           Enter interactive review mode after checking")
        print("  --review-threshold=N    Set review threshold for low-confidence items")
        print("  --show-metrics          Display performance metrics after check")
        print("  --use-multiagent        Explicitly enable multi-agent mode (default)")
        print("  --legacy-output         Use legacy JSON output format")
        print("\nExamples:")
        print(f"  python {program_name} exemple.json")
        print(f"  python {program_name} exemple.json --hybrid-mode=on")
        print(f"  python {program_name} exemple.json --show-metrics")
        print(f"  python {program_name} exemple.json --review-mode")
        print("\nFeatures:")
        print("  - Multi-agent architecture with specialized compliance agents")
        print("  - Parallel execution of independent checks (30% faster)")
        print("  - State persistence and workflow resumability")
        print("  - Human-in-the-loop integration with review queue")
        print("  - AI-enhanced semantic understanding")
        print("  - Context-aware false positive elimination")
        print("  - Backward compatible with check.py interface")
        print("  - Same JSON output format as check.py")
        print("\nThe JSON file should contain:")
        print("  - document_metadata with fund_isin, client_type, etc.")
        print("  - page_de_garde, slide_2, pages_suivantes, page_de_fin")
        print("="*70)


def create_compatibility_layer() -> CompatibilityLayer:
    """
    Factory function to create a compatibility layer instance
    
    Returns:
        CompatibilityLayer instance
    """
    return CompatibilityLayer()


# Convenience functions for common operations

def translate_args(args: List[str]) -> Dict[str, Any]:
    """Translate command-line arguments"""
    layer = create_compatibility_layer()
    return layer.translate_command_line_args(args)


def ensure_compatible_output(output: Dict[str, Any], include_agent_metadata: bool = False) -> Dict[str, Any]:
    """Ensure output is compatible with existing tools"""
    layer = create_compatibility_layer()
    return layer.ensure_output_compatibility(output, include_agent_metadata)


def normalize_violations(violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize violations for compatibility"""
    layer = create_compatibility_layer()
    return layer.normalize_violations(violations)
