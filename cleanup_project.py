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
    # Terminal Output Files - will be detected dynamically
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
    
    # Backup Files
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
    
    # Redundant Documentation
    "TASK5_IMPLEMENTATION_SUMMARY.md",
    "TASK6_IMPLEMENTATION_SUMMARY.md",
    "TASK7_AUDIT_TRAIL_SUMMARY.md",
    "TASK8_IMPLEMENTATION_SUMMARY.md",
    "TASK9_IMPLEMENTATION_SUMMARY.md",
    "BATCH_OPERATIONS_IMPLEMENTATION.md",
    "BATCH_OPERATIONS_QUICK_REFERENCE.md",
    "BATCH_REVIEW_GUIDE.md",
    "HITL_CONFIGURATION_GUIDE.md",
    "HITL_CONFIG_QUICK_REFERENCE.md",
    "HITL_INTEGRATION.md",
    "FEEDBACK_INTEGRATION_GUIDE.md",
    "METRICS_USAGE_GUIDE.md",
    "PERSISTENCE_DOCUMENTATION.md",
    "REVIEW_MODE_USAGE.md",
    
    # Test Scripts (keep only essential ones)
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
]

# Directories to remove
DIRS_TO_REMOVE = [
    "__pycache__",
    ".pytest_cache",
    "test_audit_integration",
    "test_audit_logs",
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
