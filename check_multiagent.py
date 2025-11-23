#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Agent Compliance Checker Entry Point

This is the main entry point for the LangGraph-based multi-agent compliance system.
It provides the same command-line interface as check.py while using the new
multi-agent architecture for improved modularity, parallel processing, and HITL integration.

Usage: python check_multiagent.py <json_file> [options]

Features:
- Multi-agent architecture with specialized compliance agents
- Parallel execution of independent checks (30% faster)
- State persistence and workflow resumability
- Human-in-the-loop integration with review queue
- Backward compatible with existing check.py interface
- Same JSON output format as check.py
"""

import sys
import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid

# Fix encoding for ALL output streams
if sys.platform == 'win32':
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass
    
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
else:
    # Unix/Linux
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Set environment encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

from dotenv import load_dotenv

# Load environment
load_dotenv()

if not os.path.exists('.env'):
    print("⚠️  Warning: .env file not found. Some features may be limited.")

# Import compatibility layer
try:
    from compatibility_layer import create_compatibility_layer
    COMPATIBILITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Compatibility layer not available: {e}")
    COMPATIBILITY_AVAILABLE = False


class TeeOutput:
    """Captures output to both console and file"""
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'w', encoding='utf-8', errors='replace')
        self.original = original_stream
        
    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.file.flush()
        
    def flush(self):
        self.original.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store HITL components for metrics display
_HITL_COMPONENTS = {}


def load_configuration(config_path: str = "hybrid_config.json") -> Dict[str, Any]:
    """
    Load configuration from hybrid_config.json
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✓ Configuration loaded from {config_path}")
            return config
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}


def parse_command_line_args(args: list) -> Dict[str, Any]:
    """
    Parse command-line arguments (same as check.py)
    
    Uses compatibility layer for consistent argument parsing
    
    Args:
        args: Command-line arguments (sys.argv)
        
    Returns:
        Dictionary of parsed options
    """
    # Use compatibility layer if available
    if COMPATIBILITY_AVAILABLE:
        compat_layer = create_compatibility_layer()
        options = compat_layer.translate_command_line_args(args)
        # Ensure use_multiagent is always true for this entry point
        options['use_multiagent'] = True
        return options
    
    # Fallback to manual parsing
    options = {
        'json_file': None,
        'hybrid_mode': None,
        'rules_only': False,
        'context_aware': None,
        'ai_confidence': None,
        'review_mode': False,
        'review_threshold': None,
        'show_metrics': False,
        'use_multiagent': True,  # Always true for this entry point
        'legacy_output': False,
        'test_hitl': False  # Test HITL integration
    }
    
    if len(args) < 2:
        return options
    
    options['json_file'] = args[1]
    
    # Parse additional options
    for arg in args[2:]:
        if arg.startswith('--hybrid-mode='):
            mode = arg.split('=')[1].lower()
            options['hybrid_mode'] = (mode == 'on')
        elif arg == '--rules-only':
            options['rules_only'] = True
        elif arg.startswith('--context-aware='):
            mode = arg.split('=')[1].lower()
            options['context_aware'] = (mode == 'on')
        elif arg.startswith('--ai-confidence='):
            options['ai_confidence'] = int(arg.split('=')[1])
        elif arg == '--review-mode':
            options['review_mode'] = True
        elif arg.startswith('--review-threshold='):
            options['review_threshold'] = int(arg.split('=')[1])
        elif arg == '--show-metrics':
            options['show_metrics'] = True
        elif arg == '--legacy-output':
            options['legacy_output'] = True
        elif arg == '--test-hitl':
            options['test_hitl'] = True
    
    return options


def apply_command_line_config(config: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply command-line options to configuration
    
    Uses compatibility layer for consistent configuration mapping
    
    Args:
        config: Base configuration dictionary
        options: Parsed command-line options
        
    Returns:
        Updated configuration dictionary
    """
    # Use compatibility layer if available
    if COMPATIBILITY_AVAILABLE:
        compat_layer = create_compatibility_layer()
        config = compat_layer.map_config_options(options, config)
    else:
        # Fallback to manual mapping
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
        if options.get('legacy_output', False):
            config['use_legacy_format'] = True
    
    # Print configuration changes
    if options['hybrid_mode'] is not None:
        if options['hybrid_mode']:
            print("✓ Hybrid mode enabled via command line")
        else:
            print("✓ Hybrid mode disabled via command line")
    
    if options['rules_only']:
        print("✓ Rules-only mode enabled via command line")
    
    if options['context_aware'] is not None:
        if options['context_aware']:
            print("✓ AI Context-Aware mode enabled via command line")
            print("  This mode uses semantic understanding to eliminate false positives")
        else:
            print("✓ AI Context-Aware mode disabled via command line")
    
    if options['ai_confidence'] is not None:
        print(f"✓ AI confidence threshold set to {options['ai_confidence']}%")
    
    if options['review_threshold'] is not None:
        print(f"✓ Review threshold set to {options['review_threshold']}%")
    
    if options['review_mode']:
        print("✓ Interactive review mode enabled")
    
    if options.get('legacy_output', False):
        print("✓ Legacy output format enabled")
    
    return config


def initialize_workflow(config: Dict[str, Any]) -> Any:
    """
    Initialize the LangGraph multi-agent workflow with HITL integration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled LangGraph workflow
    """
    try:
        from workflow_builder import create_compliance_workflow
        
        # Enable checkpointing for state persistence
        enable_checkpointing = config.get('multi_agent', {}).get('state_persistence', True)
        checkpoint_db_path = config.get('multi_agent', {}).get('checkpoint_db_path', 'checkpoints/compliance_workflow.db')
        
        logger.info("Initializing multi-agent workflow with HITL integration...")
        
        # Initialize HITL components
        from review_manager import ReviewManager
        from audit_logger import AuditLogger
        from feedback_loop import FeedbackInterface, PatternAnalyzer, LearningEngine
        from confidence_calibrator import ConfidenceCalibrator
        from feedback_integration import FeedbackIntegration
        
        # Initialize audit logger
        audit_dir = config.get('hitl', {}).get('audit_dir', './audit_logs/')
        audit_logger = AuditLogger(audit_dir=audit_dir)
        logger.info(f"✓ Audit logger initialized (dir: {audit_dir})")
        
        # Initialize feedback components
        feedback_interface = FeedbackInterface(db_path="feedback_data.json")
        confidence_calibrator = ConfidenceCalibrator(db_path="calibration_data.json")
        pattern_analyzer = PatternAnalyzer(feedback_interface)
        
        # Initialize feedback integration for real-time learning
        feedback_integration = FeedbackIntegration(
            feedback_interface=feedback_interface,
            confidence_calibrator=confidence_calibrator,
            pattern_detector=None,  # Optional
            audit_logger=audit_logger
        )
        logger.info("✓ Feedback integration initialized with real-time learning")
        
        # Initialize review manager with feedback integration and audit logger
        review_manager = ReviewManager(
            queue_file="review_queue.json",
            feedback_integration=feedback_integration,
            audit_logger=audit_logger
        )
        logger.info("✓ Review manager initialized")
        
        # Store HITL components in global variable for metrics display
        global _HITL_COMPONENTS
        _HITL_COMPONENTS = {
            'review_manager': review_manager,
            'audit_logger': audit_logger,
            'feedback_interface': feedback_interface,
            'confidence_calibrator': confidence_calibrator,
            'feedback_integration': feedback_integration,
            'pattern_analyzer': pattern_analyzer
        }
        
        # Register components in global registry for workflow_builder access
        import hitl_registry
        hitl_registry.register_component('review_manager', review_manager)
        hitl_registry.register_component('audit_logger', audit_logger)
        hitl_registry.register_component('feedback_interface', feedback_interface)
        hitl_registry.register_component('confidence_calibrator', confidence_calibrator)
        hitl_registry.register_component('feedback_integration', feedback_integration)
        
        # Add a flag to config to indicate HITL components are available in registry
        config['hitl_registry_enabled'] = True
        
        # Create workflow (it will use the global registry to access HITL components)
        workflow = create_compliance_workflow(
            config=config,
            enable_checkpointing=enable_checkpointing,
            checkpoint_db_path=checkpoint_db_path
        )
        
        logger.info("✓ Multi-agent workflow initialized successfully with HITL integration")
        return workflow
        
    except Exception as e:
        logger.error(f"Failed to initialize workflow: {e}")
        raise


def execute_compliance_check(
    workflow: Any,
    json_file_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute compliance checking using the multi-agent workflow
    
    Args:
        workflow: Compiled LangGraph workflow
        json_file_path: Path to JSON document
        config: Configuration dictionary
        
    Returns:
        Dictionary with violations and metadata
    """
    try:
        # Load document
        with open(json_file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        
        # Extract metadata
        doc_metadata = doc.get('document_metadata', {})
        fund_isin = doc_metadata.get('fund_isin')
        client_type = doc_metadata.get('client_type', 'retail')
        doc_type = doc_metadata.get('document_type', 'fund_presentation')
        fund_status = doc_metadata.get('fund_status', 'active')
        esg_classification = doc_metadata.get('fund_esg_classification', 'other')
        
        # Print header
        print(f"\n{'='*70}")
        print(f"📋 COMPLIANCE REPORT - MULTI-AGENT VERSION")
        print(f"{'='*70}")
        print(f"File: {json_file_path}")
        print(f"Fund ISIN: {fund_isin or 'Not specified'}")
        print(f"Client Type: {client_type.upper()}")
        print(f"Document Type: {doc_type}")
        print(f"Fund Status: {fund_status}")
        print(f"ESG Classification: {esg_classification}")
        print(f"{'='*70}\n")
        
        # Initialize state from data_models_multiagent
        from data_models_multiagent import initialize_compliance_state
        
        initial_state = initialize_compliance_state(
            document=doc,
            document_id=json_file_path,
            config=config
        )
        
        # Generate unique thread ID for this check
        thread_id = f"check_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Execute workflow
        logger.info(f"Executing workflow with thread_id: {thread_id}")
        print("🚀 Starting multi-agent compliance workflow...\n")
        
        # Invoke workflow with checkpointing
        final_state = workflow.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract results
        violations = final_state.get('violations', [])
        workflow_status = final_state.get('workflow_status', 'unknown')
        agent_timings = final_state.get('agent_timings', {})
        
        logger.info(f"Workflow completed with status: {workflow_status}")
        logger.info(f"Total violations found: {len(violations)}")
        
        # Print final report
        print(f"\n{'='*70}")
        if len(violations) == 0:
            print("✅ NO VIOLATIONS FOUND - Document is compliant!")
        else:
            print(f"❌ {len(violations)} VIOLATION(S) FOUND")
        print(f"{'='*70}\n")
        
        # Display violations
        for i, v in enumerate(violations, 1):
            print(f"{'='*70}")
            print(f"[{v.get('severity', 'UNKNOWN')}] {v.get('type', 'UNKNOWN')} Violation #{i}")
            print(f"{'='*70}")
            print(f"› Rule: {v.get('rule', 'N/A')}")
            print(f"› Issue: {v.get('message', 'N/A')}")
            print(f"  Location: {v.get('slide', 'N/A')} - {v.get('location', 'N/A')}")
            print(f"\n› Evidence:")
            print(f"   {v.get('evidence', 'N/A')}")
            print(f"  Confidence: {v.get('confidence', 'N/A')}%")
            
            # Display AI reasoning if available
            if 'ai_reasoning' in v and v['ai_reasoning']:
                print(f"\n🤖 AI Reasoning:")
                print(f"   {v['ai_reasoning']}")
            
            # Display method used
            if 'method' in v and v['method']:
                print(f"  Method: {v['method']}")
            
            # Display agent that detected this violation
            if 'detected_by' in v and v['detected_by']:
                print(f"  Detected by: {v['detected_by']}")
            
            print()
        
        # Summary
        if violations:
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")
            
            # By type
            type_counts = {}
            for v in violations:
                vtype = v.get('type', 'UNKNOWN')
                type_counts[vtype] = type_counts.get(vtype, 0) + 1
            
            print(f"\nViolations by type:")
            for vtype, count in sorted(type_counts.items()):
                print(f"   {vtype}: {count}")
            
            # By severity
            severity_counts = {}
            for v in violations:
                sev = v.get('severity', 'UNKNOWN')
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            print(f"\nViolations by severity:")
            for sev in ['CRITICAL', 'MAJOR', 'WARNING']:
                if sev in severity_counts:
                    print(f"   {sev}: {severity_counts[sev]}")
        
        # Display agent execution times
        if agent_timings:
            print(f"\n{'='*70}")
            print(f"AGENT EXECUTION TIMES")
            print(f"{'='*70}")
            total_time = sum(agent_timings.values())
            for agent_name, duration in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True):
                print(f"  {agent_name}: {duration:.2f}s")
            print(f"\n  Total: {total_time:.2f}s")
            print(f"{'='*70}")
        
        return {
            'violations': violations,
            'total_violations': len(violations),
            'metadata': doc_metadata,
            'workflow_status': workflow_status,
            'agent_timings': agent_timings,
            'thread_id': thread_id,
            'queued_for_review': len(final_state.get('review_queue', [])),
            'queue_statistics': final_state.get('queue_statistics', {}),
            'batch_opportunities': final_state.get('batch_opportunities', [])
        }
        
    except Exception as e:
        logger.error(f"Error during compliance check: {e}", exc_info=True)
        return {
            'error': str(e),
            'violations': [],
            'total_violations': 0
        }


def generate_json_output(
    result: Dict[str, Any],
    json_file_path: str,
    config: Dict[str, Any]
) -> str:
    """
    Generate JSON output in same format as check.py
    
    Uses compatibility layer to ensure backward compatibility
    
    Args:
        result: Compliance check result
        json_file_path: Original JSON file path
        config: Configuration dictionary
        
    Returns:
        Path to output file
    """
    try:
        from output_formatter import create_formatter
        
        # Get formatter configuration
        formatter_config = {
            'use_legacy_format': config.get('use_legacy_format', False),
            'ai_enabled': config.get('ai_enabled', False)
        }
        
        formatter = create_formatter(formatter_config)
        
        # Normalize violations using compatibility layer
        violations = result.get('violations', [])
        if COMPATIBILITY_AVAILABLE:
            compat_layer = create_compatibility_layer()
            compat_layer.legacy_mode = formatter_config['use_legacy_format']
            violations = compat_layer.normalize_violations(violations)
        
        # Format complete output
        output = formatter.format_complete_output(
            json_file_path=json_file_path,
            metadata=result.get('metadata', {}),
            violations=violations
        )
        
        # Add multi-agent specific metadata (unless in legacy mode)
        include_agent_metadata = not formatter.legacy_mode
        if include_agent_metadata:
            output['multi_agent'] = {
                'enabled': True,
                'workflow_status': result.get('workflow_status', 'unknown'),
                'agent_timings': result.get('agent_timings', {}),
                'thread_id': result.get('thread_id', ''),
                'total_execution_time': sum(result.get('agent_timings', {}).values())
            }
        
        # Ensure output compatibility using compatibility layer
        if COMPATIBILITY_AVAILABLE:
            output = compat_layer.ensure_output_compatibility(output, include_agent_metadata)
        
        # Validate output structure
        if not formatter.validate_output_structure(output):
            logger.warning("Output structure validation failed")
        
        # Save JSON output
        output_filename = json_file_path.replace('.json', '_violations.json')
        formatter.save_output(output, output_filename)
        
        print(f"\n{'='*70}")
        print(f"📄 JSON output saved to: {output_filename}")
        if not formatter.legacy_mode and formatter.include_ai_fields:
            print(f"   Format: Enhanced (with AI and multi-agent fields)")
        else:
            print(f"   Format: Legacy (backward compatible)")
        print(f"{'='*70}\n")
        
        return output_filename
        
    except Exception as e:
        logger.error(f"Error generating JSON output: {e}")
        # Fallback: save basic JSON with compatibility layer
        output_filename = json_file_path.replace('.json', '_violations.json')
        
        basic_output = {
            'document_id': json_file_path,
            'violations': result.get('violations', []),
            'total_violations': result.get('total_violations', 0),
            'metadata': result.get('metadata', {})
        }
        
        # Apply compatibility layer to fallback output
        if COMPATIBILITY_AVAILABLE:
            compat_layer = create_compatibility_layer()
            basic_output = compat_layer.ensure_output_compatibility(basic_output, False)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(basic_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 JSON output saved to: {output_filename} (basic format)")
        return output_filename


def display_metrics(result: Dict[str, Any], config: Dict[str, Any]):
    """
    Display performance metrics including HITL integration status
    
    Args:
        result: Compliance check result
        config: Configuration dictionary
    """
    print(f"\n{'='*70}")
    print("📊 PERFORMANCE METRICS")
    print(f"{'='*70}")
    
    # Agent execution times
    agent_timings = result.get('agent_timings', {})
    if agent_timings:
        print(f"\nAgent Execution Times:")
        total_time = sum(agent_timings.values())
        for agent_name, duration in sorted(agent_timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {agent_name}: {duration:.2f}s ({percentage:.1f}%)")
        print(f"\n  Total: {total_time:.2f}s")
    
    # Workflow status
    workflow_status = result.get('workflow_status', 'unknown')
    print(f"\nWorkflow Status: {workflow_status}")
    
    # Review queue
    queued_count = result.get('queued_for_review', 0)
    if queued_count > 0:
        print(f"\nReview Queue: {queued_count} item(s) queued for human review")
        
        # Display queue statistics if available
        if 'queue_statistics' in result:
            stats = result['queue_statistics']
            print(f"  Total pending: {stats.get('total_pending', 0)}")
            print(f"  Total in review: {stats.get('total_in_review', 0)}")
            print(f"  Total reviewed: {stats.get('total_reviewed', 0)}")
            print(f"  Average confidence: {stats.get('avg_confidence', 0):.1f}%")
    
    # HITL integration status (stored in global variable to avoid serialization issues)
    global _HITL_COMPONENTS
    hitl_components = _HITL_COMPONENTS
    if hitl_components:
        print(f"\n{'='*70}")
        print("🔗 HITL INTEGRATION STATUS")
        print(f"{'='*70}")
        
        # Feedback integration stats
        feedback_integration = hitl_components.get('feedback_integration')
        if feedback_integration:
            processing_stats = feedback_integration.get_processing_stats()
            print(f"\nFeedback Processing:")
            print(f"  Total processed: {processing_stats['total_processed']}")
            if processing_stats['total_processed'] > 0:
                print(f"  Avg processing time: {processing_stats['avg_processing_time_ms']:.1f}ms")
                print(f"  Under 1s rate: {processing_stats['under_1s_rate']:.1%}")
        
        # Confidence calibration status
        confidence_calibrator = hitl_components.get('confidence_calibrator')
        if confidence_calibrator:
            print(f"\nConfidence Calibration:")
            print(f"  Total records: {len(confidence_calibrator.records)}")
            if len(confidence_calibrator.records) > 0:
                overall_metrics = confidence_calibrator.get_calibration_metrics()
                print(f"  Overall accuracy: {overall_metrics.accuracy:.1%}")
                print(f"  Calibration score: {overall_metrics.calibration_score:.3f}")
        
        # Audit logging status
        audit_logger = hitl_components.get('audit_logger')
        if audit_logger:
            print(f"\nAudit Trail:")
            print(f"  Total entries: {len(audit_logger.audit_entries)}")
            integrity = audit_logger.verify_integrity()
            print(f"  Integrity: {'✓ Valid' if integrity['valid'] else '✗ Invalid'}")
    
    print(f"{'='*70}")


def test_hitl_integration(config: Dict[str, Any]) -> bool:
    """
    Test HITL integration components
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all tests pass
    """
    print(f"\n{'='*70}")
    print("🧪 TESTING HITL INTEGRATION")
    print(f"{'='*70}\n")
    
    global _HITL_COMPONENTS
    hitl_components = _HITL_COMPONENTS
    if not hitl_components:
        print("⚠️  No HITL components found in configuration")
        return False
    
    all_tests_passed = True
    
    # Test 1: Review Manager
    print("1. Testing Review Manager...")
    review_manager = hitl_components.get('review_manager')
    if review_manager:
        try:
            stats = review_manager.get_queue_stats()
            print(f"   ✓ Review Manager operational")
            print(f"     - Pending: {stats.total_pending}")
            print(f"     - In Review: {stats.total_in_review}")
            print(f"     - Reviewed: {stats.total_reviewed}")
        except Exception as e:
            print(f"   ✗ Review Manager test failed: {e}")
            all_tests_passed = False
    else:
        print("   ✗ Review Manager not initialized")
        all_tests_passed = False
    
    # Test 2: Audit Logger
    print("\n2. Testing Audit Logger...")
    audit_logger = hitl_components.get('audit_logger')
    if audit_logger:
        try:
            integrity = audit_logger.verify_integrity()
            print(f"   ✓ Audit Logger operational")
            print(f"     - Total entries: {integrity['total_entries']}")
            print(f"     - Integrity: {'Valid' if integrity['valid'] else 'Invalid'}")
        except Exception as e:
            print(f"   ✗ Audit Logger test failed: {e}")
            all_tests_passed = False
    else:
        print("   ✗ Audit Logger not initialized")
        all_tests_passed = False
    
    # Test 3: Feedback Interface
    print("\n3. Testing Feedback Interface...")
    feedback_interface = hitl_components.get('feedback_interface')
    if feedback_interface:
        try:
            pending = feedback_interface.get_pending_reviews()
            history = feedback_interface.get_feedback_history(days=30)
            print(f"   ✓ Feedback Interface operational")
            print(f"     - Pending reviews: {len(pending)}")
            print(f"     - Historical records (30 days): {len(history)}")
        except Exception as e:
            print(f"   ✗ Feedback Interface test failed: {e}")
            all_tests_passed = False
    else:
        print("   ✗ Feedback Interface not initialized")
        all_tests_passed = False
    
    # Test 4: Confidence Calibrator
    print("\n4. Testing Confidence Calibrator...")
    confidence_calibrator = hitl_components.get('confidence_calibrator')
    if confidence_calibrator:
        try:
            metrics = confidence_calibrator.get_calibration_metrics()
            print(f"   ✓ Confidence Calibrator operational")
            print(f"     - Total predictions: {metrics.total_predictions}")
            if metrics.total_predictions > 0:
                print(f"     - Accuracy: {metrics.accuracy:.1%}")
                print(f"     - Calibration score: {metrics.calibration_score:.3f}")
        except Exception as e:
            print(f"   ✗ Confidence Calibrator test failed: {e}")
            all_tests_passed = False
    else:
        print("   ✗ Confidence Calibrator not initialized")
        all_tests_passed = False
    
    # Test 5: Feedback Integration
    print("\n5. Testing Feedback Integration...")
    feedback_integration = hitl_components.get('feedback_integration')
    if feedback_integration:
        try:
            processing_stats = feedback_integration.get_processing_stats()
            accuracy_metrics = feedback_integration.get_accuracy_metrics()
            print(f"   ✓ Feedback Integration operational")
            print(f"     - Total processed: {processing_stats['total_processed']}")
            print(f"     - Total reviews: {accuracy_metrics['total_reviews']}")
            if accuracy_metrics['total_reviews'] > 0:
                print(f"     - Accuracy: {accuracy_metrics['accuracy']:.1%}")
        except Exception as e:
            print(f"   ✗ Feedback Integration test failed: {e}")
            all_tests_passed = False
    else:
        print("   ✗ Feedback Integration not initialized")
        all_tests_passed = False
    
    # Test 6: Review CLI compatibility
    print("\n6. Testing Review CLI compatibility...")
    try:
        from review import ReviewCLI
        # Try to instantiate CLI with the review manager
        cli = ReviewCLI(
            reviewer_id="test_reviewer",
            queue_file="review_queue.json",
            audit_dir="./audit_logs/"
        )
        print(f"   ✓ Review CLI compatible")
        print(f"     - CLI can be instantiated")
        print(f"     - Queue file: review_queue.json")
        print(f"     - Audit dir: ./audit_logs/")
    except Exception as e:
        print(f"   ✗ Review CLI test failed: {e}")
        all_tests_passed = False
    
    print(f"\n{'='*70}")
    if all_tests_passed:
        print("✅ ALL HITL INTEGRATION TESTS PASSED")
    else:
        print("⚠️  SOME HITL INTEGRATION TESTS FAILED")
    print(f"{'='*70}\n")
    
    return all_tests_passed


def main():
    """Main function"""
    # Parse command-line arguments
    options = parse_command_line_args(sys.argv)
    
    if options['json_file'] is None:
        # Use compatibility layer for consistent usage display
        if COMPATIBILITY_AVAILABLE:
            compat_layer = create_compatibility_layer()
            compat_layer.print_usage("check_multiagent.py")
        else:
            # Fallback usage display
            print("\n" + "="*70)
            print("MULTI-AGENT COMPLIANCE CHECKER")
            print("="*70)
            print("\nUsage:")
            print("  python check_multiagent.py <json_file> [options]")
            print("\nOptions:")
            print("  --hybrid-mode=on|off    Enable/disable AI+Rules hybrid mode")
            print("  --rules-only            Use only rule-based checking (no AI)")
            print("  --context-aware=on|off  Enable/disable AI context-aware mode")
            print("  --ai-confidence=N       Set AI confidence threshold (default: 70)")
            print("  --review-mode           Enter interactive review mode after checking")
            print("  --review-threshold=N    Set review threshold for low-confidence items")
            print("  --show-metrics          Display performance metrics after check")
            print("  --legacy-output         Use legacy JSON output format")
            print("  --test-hitl             Test HITL integration before running check")
            print("\nExamples:")
            print("  python check_multiagent.py exemple.json")
            print("  python check_multiagent.py exemple.json --hybrid-mode=on")
            print("  python check_multiagent.py exemple.json --show-metrics")
            print("  python check_multiagent.py exemple.json --review-mode")
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
        sys.exit(1)
    
    json_file = options['json_file']
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"\n❌ File '{json_file}' not found")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📄 Available JSON files:")
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            for f in json_files:
                print(f"  - {f}")
        else:
            print("  (none found)")
        sys.exit(1)
    
    # Set up terminal output logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    terminal_log_file = f"terminal_output_multiagent_{timestamp}.txt"
    
    # Redirect stdout and stderr to both console and file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    sys.stdout = TeeOutput(terminal_log_file, original_stdout)
    sys.stderr = TeeOutput(terminal_log_file, original_stderr)
    
    print(f"{'='*70}")
    print(f"Terminal output will be saved to: {terminal_log_file}")
    print(f"{'='*70}\n")
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Apply command-line options
        config = apply_command_line_config(config, options)
        
        # Initialize workflow
        workflow = initialize_workflow(config)
        
        # Test HITL integration if requested
        if options.get('test_hitl', False):
            print(f"\n{'='*70}")
            print("🧪 HITL INTEGRATION TEST MODE")
            print(f"{'='*70}\n")
            test_hitl_integration(config)
            print("\nContinuing with compliance check...\n")
        
        # Execute compliance check
        result = execute_compliance_check(workflow, json_file, config)
        
        if 'error' not in result:
            # Generate JSON output
            output_file = generate_json_output(result, json_file, config)
            
            print(f"\n{'='*70}")
            print("✅ CHECK COMPLETE")
            print(f"{'='*70}")
            
            if result['total_violations'] == 0:
                print("\n🎉 Document is fully compliant!")
            else:
                print(f"\n⚠️  Please review and fix {result['total_violations']} violation(s)")
                print(f"📄 Detailed JSON report: {output_file}")
            
            # Display performance metrics if requested
            if options['show_metrics']:
                display_metrics(result, config)
            
            # Display review mode option if items were queued
            if result.get('queued_for_review', 0) > 0:
                print(f"\n{'='*70}")
                print(f"👤 HUMAN REVIEW REQUIRED")
                print(f"{'='*70}")
                print(f"\n{result['queued_for_review']} violation(s) queued for human review")
                print(f"Run 'python review.py' to start reviewing")
                
                # Display queue statistics
                if 'queue_statistics' in result:
                    stats = result['queue_statistics']
                    print(f"\nQueue Statistics:")
                    print(f"  Total pending: {stats.get('total_pending', 0)}")
                    print(f"  Average confidence: {stats.get('avg_confidence', 0):.1f}%")
                    if stats.get('by_severity'):
                        print(f"  By severity: {stats['by_severity']}")
                
                print(f"{'='*70}\n")
                
                # Enter review mode if requested
                if options['review_mode']:
                    print("\n🔍 Entering interactive review mode...")
                    print("   Press Ctrl+C to exit\n")
                    
                    try:
                        from review import ReviewCLI
                        # Pass the review manager from the workflow
                        cli = ReviewCLI(
                            reviewer_id="cli_user",
                            queue_file="review_queue.json",
                            audit_dir="./audit_logs/"
                        )
                        cli.start_interactive_session()
                    except ImportError:
                        print("⚠️  Review CLI not available. Please run 'python review.py' separately.")
                    except KeyboardInterrupt:
                        print("\n\n✓ Review mode cancelled by user")
                    except Exception as e:
                        print(f"\n⚠️  Error starting review mode: {e}")
                        print("   Please run 'python review.py' separately.")
            
            # Close terminal log files
            print(f"\n{'='*70}")
            print(f"📄 Terminal output saved to: {terminal_log_file}")
            print(f"{'='*70}")
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            if result['total_violations'] == 0:
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print(f"\n❌ Error during compliance check: {result['error']}")
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        
        # Close terminal log files on error
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure terminal log is closed even on unexpected errors
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()
            sys.stderr.close()
        raise
