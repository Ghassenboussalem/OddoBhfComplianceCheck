#!/usr/bin/env python3
"""
Project Cleanup Script
Removes redundant, obsolete, and example files from the project.
"""

import os
import shutil
from pathlib import Path

# Files to remove
FILES_TO_REMOVE = [
    # Terminal Output Files - November 22
    "terminal_output_20251122_114645.txt",
    "terminal_output_20251122_114929.txt",
    "terminal_output_20251122_115204.txt",
    "terminal_output_20251122_115251.txt",
    "terminal_output_20251122_115638.txt",
    "terminal_output_20251122_121627.txt",
    "terminal_output_20251122_122811.txt",
    "terminal_output_20251122_122920.txt",
    "terminal_output_20251122_123016.txt",
    "terminal_output_20251122_123102.txt",
    "terminal_output_20251122_123408.txt",
    "terminal_output_20251122_123711.txt",
    "terminal_output_20251122_124038.txt",
    "terminal_output_20251122_124340.txt",
    "terminal_output_20251122_124646.txt",
    "terminal_output_20251122_155657.txt",
    "terminal_output_20251122_163224.txt",
    
    # Terminal Output Files - November 23
    "terminal_output_20251123_102345.txt",
    "terminal_output_20251123_102835.txt",
    "terminal_output_20251123_104718.txt",
    "terminal_output_multiagent_20251123_091756.txt",
    "terminal_output_multiagent_20251123_091906.txt",
    "terminal_output_multiagent_20251123_092722.txt",
    "terminal_output_multiagent_20251123_092727.txt",
    "terminal_output_multiagent_20251123_092732.txt",
    "terminal_output_multiagent_20251123_092737.txt",
    "terminal_output_multiagent_20251123_092742.txt",
    "terminal_output_multiagent_20251123_092747.txt",
    "terminal_output_multiagent_20251123_092753.txt",
    "terminal_output_multiagent_20251123_092758.txt",
    "terminal_output_multiagent_20251123_092803.txt",
    "terminal_output_multiagent_20251123_092808.txt",
    "terminal_output_multiagent_20251123_092813.txt",
    "terminal_output_multiagent_20251123_092818.txt",
    "terminal_output_multiagent_20251123_102618.txt",
    "terminal_output_multiagent_20251123_102737.txt",
    "terminal_output_multiagent_20251123_103213.txt",
    "terminal_output_multiagent_20251123_104624.txt",
    "terminal_output_multiagent_20251123_121402.txt",
    "terminal_output_multiagent_20251123_121443.txt",
    "terminal_output_multiagent_20251123_121550.txt",
    "terminal_output_multiagent_20251123_121819.txt",
    "terminal_output_multiagent_20251123_121941.txt",
    "terminal_output_multiagent_20251123_124934.txt",
    "terminal_output_multiagent_20251123_124954.txt",
    "terminal_output_multiagent_20251123_125025.txt",
    "terminal_output_multiagent_20251123_125048.txt",
    "terminal_output_multiagent_20251123_125107.txt",
    "terminal_output_multiagent_20251123_125128.txt",
    "terminal_output_multiagent_20251123_125148.txt",
    "terminal_output_multiagent_20251123_125208.txt",
    "terminal_output_multiagent_20251123_125231.txt",
    "terminal_output_multiagent_20251123_125251.txt",
    "terminal_output_multiagent_20251123_125310.txt",
    "terminal_output_multiagent_20251123_125330.txt",
    "terminal_output_multiagent_20251123_125351.txt",
    "terminal_output_multiagent_20251123_125411.txt",
    "terminal_output_multiagent_20251123_125432.txt",
    "terminal_output_multiagent_20251123_125453.txt",
    "terminal_output_multiagent_20251123_125506.txt",
    "terminal_output_multiagent_20251123_125525.txt",
    "terminal_output_multiagent_20251123_125545.txt",
    "terminal_output_multiagent_20251123_125609.txt",
    "terminal_output_multiagent_20251123_125648.txt",
    
    # Backup Files - November 21
    "review_queue.json.backup.20251121_212910",
    "review_queue.json.backup.20251121_212959",
    "review_queue.json.backup.20251121_213015",
    "review_queue.json.backup.20251121_213056",
    "review_queue.json.backup.20251121_213116",
    "test_review_queue.json.backup.20251121_202418",
    "test_persistence.json.test.20251121_202201",
    "audit_logs/audit_log.json.backup.20251121_213015",
    "audit_logs/audit_log.json.backup.20251121_213056",
    "audit_logs/audit_log.json.backup.20251121_213116",
    "test_audit_logs/audit_log.json.backup.20251121_202520",
    
    # Backup Files - November 22-23
    "review_queue.json.backup.20251122_205028",
    "review_queue.json.backup.20251123_100823",
    "review_queue.json.backup.20251123_100824",
    "review_queue.json.backup.20251123_100830",
    "review_queue.json.backup.20251123_100943",
    "review_queue.json.backup.20251123_100944",
    "review_queue.json.backup.20251123_100949",
    "review_queue.json.backup.20251123_100950",
    "review_queue.json.backup.20251123_101649",
    "review_queue.json.backup.20251123_101810",
    "review_queue.json.backup.20251123_101938",
    "review_queue.json.backup.20251123_102056",
    "review_queue.json.backup.20251123_125522",
    "review_queue.json.backup.20251123_125523",
    "review_queue.json.backup.20251123_125543",
    "test_review_tools_queue.json.backup.20251122_202539",
    "test_review_tools_queue.json.backup.20251122_202540",
    "test_reviewer_agent_queue.json.backup.20251122_203039",
    "test_reviewer_agent_queue.json.backup.20251122_203040",
    "test_hitl_review_queue.json.backup.20251123_121355",
    
    # Old Test/Result Files
    "test_audit_calibration.json",
    "test_audit_export.csv",
    "test_audit_export.json",
    "test_audit_feedback.json",
    "test_audit_integration_export.csv",
    "test_audit_integration_export.json",
    "test_audit_integration_report.json",
    "test_audit_review_queue.json",
    "test_calibration_integration.json",
    "test_compliance_report.json",
    "test_export.json",
    "test_feedback_integration.json",
    "test_integration_metrics.json",
    "test_metrics.json",
    "test_metrics_integration.json",
    "test_metrics_report.json",
    "test_performance_report.json",
    "test_review_calibration.json",
    "test_review_feedback.json",
    "test_review_queue.json",
    "test_review_queue_integration.json",
    "test_task5_audit_export.json",
    "test_task5_calibration.json",
    "test_task5_feedback.json",
    "verify_task6_metrics.json",
    "verify_task6_report.json",
    "compliance_report_20251121_171937.json",
    "performance_report_20251121_170642.json",
    "review_metrics.json",
    "demo_review_queue.json",
    "compliance_metrics_test.json",
    "temp_perf_test_current_violations.json",
    "temp_regression_test_violations.json",
    "test_output.txt",
    "test_hitl_audit_export.json",
    "test_hitl_calibration_integration.json",
    "test_hitl_feedback_integration.json",
    "test_hitl_feedback.json",
    "test_hitl_review_queue.json",
    
    # Task Implementation Summaries (redundant - info in main docs)
    "TASK5_IMPLEMENTATION_SUMMARY.md",
    "TASK6_IMPLEMENTATION_SUMMARY.md",
    "TASK7_AUDIT_TRAIL_SUMMARY.md",
    "TASK8_IMPLEMENTATION_SUMMARY.md",
    "TASK8_COMPLETION_SUMMARY.md",
    "TASK8_INTEGRATION_GUIDE.md",
    "TASK9_IMPLEMENTATION_SUMMARY.md",
    "TASK9_COMPLETION_SUMMARY.md",
    "TASK15_VALIDATION_ANALYSIS.md",
    "TASK44_COMPLETION_SUMMARY.md",
    "TASK47_ERROR_HANDLING_SUMMARY.md",
    "TASK48_AGENT_FAILURE_RECOVERY.md",
    "TASK69_ERROR_MESSAGE_IMPROVEMENTS.md",
    "TASK70_CODE_REVIEW_SUMMARY.md",
    
    # Redundant/Duplicate Documentation
    "BATCH_OPERATIONS_IMPLEMENTATION.md",
    "BATCH_OPERATIONS_QUICK_REFERENCE.md",
    "BATCH_REVIEW_GUIDE.md",
    "HITL_CONFIGURATION_GUIDE.md",
    "HITL_CONFIG_QUICK_REFERENCE.md",
    "HITL_INTEGRATION.md",
    "FEEDBACK_INTEGRATION_GUIDE.md",
    "METRICS_USAGE_GUIDE.md",
    "METRICS_USAGE.md",
    "PERSISTENCE_DOCUMENTATION.md",
    "REVIEW_MODE_USAGE.md",
    "EVIDENCE_EXTRACTOR_IMPLEMENTATION.md",
    "ERROR_HANDLING_FLOW.md",
    "ERROR_HANDLING_USAGE_GUIDE.md",
    "PERFORMANCE_OPTIMIZATION_SUMMARY.md",
    "PERFORMANCE_QUICK_REFERENCE.md",
    "WORKFLOW_BUILDER_USAGE.md",
    "STATE_PERSISTENCE_GUIDE.md",
    "PARAMETER_COMPATIBILITY.md",
    "COMPATIBILITY_LAYER_GUIDE.md",
    
    # Task 70 Cleanup Files (temporary)
    "cleanup_code_review.py",
    "apply_code_cleanup.py",
    "code_review_report.txt",
    
    # Test Scripts (redundant - covered by tests/ directory)
    "test_batch_operations.py",
    "test_hitl_config.py",
    "test_hitl_queueing.py",
    "test_metrics_integration.py",
    "test_persistence_integration.py",
    "test_review_cli.py",
    "test_review_feedback_integration.py",
    "test_review_integration.py",
    "test_task5_requirements.py",
    "verify_task6_complete.py",
    "test_audit_integration.py",
    "demo_batch_review.py",
    "test_agent_config_manager.py",
    "test_aggregator_agent.py",
    "test_args_only.py",
    "test_base_agent.py",
    "test_compatibility_layer.py",
    "test_context_agent.py",
    "test_dashboard_simple.py",
    "test_data_models_multiagent.py",
    "test_error_handling_integration.py",
    "test_esg_tools.py",
    "test_evidence_agent.py",
    "test_evidence_extractor.py",
    "test_evidence_on_real_doc.py",
    "test_evidence_tools_integration.py",
    "test_false_positive_elimination.py",
    "test_feedback_agent.py",
    "test_general_agent.py",
    "test_hitl_integration.py",
    "test_integration_false_positives.py",
    "test_multiagent_compatibility.py",
    "test_performance_agent.py",
    "test_preprocessor_agent.py",
    "test_prospectus_agent.py",
    "test_prospectus_tools.py",
    "test_registration_agent.py",
    "test_review_tools.py",
    "test_reviewer_agent.py",
    "test_state_persistence.py",
    "test_structure_agent.py",
    "test_supervisor_agent.py",
    "test_task6.py",
    "test_task8_real.py",
    "test_task8.py",
    "test_task9.py",
    "test_tool_registry.py",
    "test_workflow_basic.py",
    "test_workflow_builder.py",
    "test_workflow_conditional_routing.py",
    "test_workflow_hitl_interrupt.py",
    "test_workflow_parallel.py",
    "test_workflow_resumability.py",
    "test_workflow_specialized.py",
    "verify_general_agent.py",
    
    # Debug/Monitoring Scripts (temporary)
    "debug_dashboard.py",
    "monitor_performance.py",
]

# Directories to remove
DIRS_TO_REMOVE = [
    "__pycache__",
    ".pytest_cache",
    "test_audit_integration",
    "test_audit_logs",
    "test_hitl_audit_logs",
    "test_metrics",
    "test_monitoring",
    "test_visualizations",
]


def remove_file(filepath):
    """Remove a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✅ Removed: {filepath}")
            return True
        else:
            print(f"⚠️  Not found: {filepath}")
            return False
    except Exception as e:
        print(f"❌ Error removing {filepath}: {e}")
        return False


def remove_directory(dirpath):
    """Remove a directory if it exists."""
    try:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            print(f"✅ Removed directory: {dirpath}")
            return True
        else:
            print(f"⚠️  Not found: {dirpath}")
            return False
    except Exception as e:
        print(f"❌ Error removing {dirpath}: {e}")
        return False


def main():
    """Main cleanup function."""
    print("=" * 70)
    print("PROJECT CLEANUP SCRIPT")
    print("=" * 70)
    print()
    
    # Get user confirmation
    print("This script will remove the following:")
    print(f"  - {len(FILES_TO_REMOVE)} redundant/obsolete files")
    print(f"  - {len(DIRS_TO_REMOVE)} cache/build directories")
    print()
    
    response = input("Do you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cleanup cancelled.")
        return
    
    print()
    print("Starting cleanup...")
    print()
    
    # Remove files
    print("Removing files...")
    print("-" * 70)
    removed_files = 0
    for filename in FILES_TO_REMOVE:
        if remove_file(filename):
            removed_files += 1
    
    print()
    
    # Remove directories
    print("Removing directories...")
    print("-" * 70)
    removed_dirs = 0
    for dirname in DIRS_TO_REMOVE:
        if remove_directory(dirname):
            removed_dirs += 1
    
    print()
    print("=" * 70)
    print("CLEANUP COMPLETE")
    print("=" * 70)
    print(f"Files removed: {removed_files}/{len(FILES_TO_REMOVE)}")
    print(f"Directories removed: {removed_dirs}/{len(DIRS_TO_REMOVE)}")
    print()
    print("✅ Project cleanup successful!")
    print()
    print("Note: Core files preserved:")
    print("  - Main implementation files (check*.py, agent.py, ai_engine.py, etc.)")
    print("  - Essential documentation (README.md, API_DOCUMENTATION.md, etc.)")
    print("  - Configuration files (*.json, .env)")
    print()


if __name__ == "__main__":
    main()
