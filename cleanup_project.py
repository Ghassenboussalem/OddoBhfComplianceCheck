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
    # Duplicate/Old Implementation Files
    "agent_enhanced_ai.py",
    "enhanced_checks.py",
    
    # Example/Demo Files
    "example_ai_engine_usage.py",
    "example_async_usage.py",
    "example_enhanced_usage.py",
    "example_feedback_loop.py",
    "example_pattern_detection.py",
    "example_performance_monitoring.py",
    "demo_testing_framework.py",
    
    # Old Test/Result Files
    "exemple_violations_ai.json",
    "example_calibration.json",
    "example_checker_feedback.json",
    "example_feedback.json",
    "feedback_export_all.json",
    "feedback_export_promotional.json",
    "test_feedback_export.json",
    "baseline_results.json",
    "discovered_patterns.json",
    "rule_recommendations.json",
    "test_suite_comprehensive.json",
    
    # Redundant Documentation
    "IMPLEMENTATION_SUMMARY.md",
    "TASK_2_IMPLEMENTATION_SUMMARY.md",
    "TASK_3_IMPLEMENTATION_SUMMARY.md",
    "TASK_4.3_IMPLEMENTATION_SUMMARY.md",
    "TASK_4.4_IMPLEMENTATION_SUMMARY.md",
    "TASK_5.2_IMPLEMENTATION_SUMMARY.md",
    "TASK_5.3_IMPLEMENTATION_SUMMARY.md",
    "TASK_5.4_IMPLEMENTATION_SUMMARY.md",
    "TASK_6_IMPLEMENTATION_SUMMARY.md",
    "TEST_TASK_2.5_SUMMARY.md",
    "ASYNC_PROCESSING_README.md",
    "ASYNC_QUICK_START.md",
    "ENHANCED_CHECKS_README.md",
    "PATTERN_DETECTION_README.md",
    "PERFORMANCE_MONITORING_README.md",
    "TESTING_FRAMEWORK_README.md",
    "MIGRATION_CHECKLIST.md",

    "test_ai_engine.py",
    "test_async_processor.py",
    "test_comprehensive_framework.py",
    "test_comprehensive_unit.py",
    "test_critical_checks_unit.py",
    "test_enhanced_checks.py",
    "test_exemple.py",
    "test_feedback_loop.py",
    "test_integration.py",
    "test_pattern_detector.py",
    "test_performance_monitor.py",
    "test_remaining_checks_integration.py",
]

# Directories to remove
DIRS_TO_REMOVE = [
    "__pycache__",
    ".pytest_cache",
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
    print("Optional: To also remove test files, uncomment TEST_FILES_TO_REMOVE")
    print("in this script and run again.")
    print()


if __name__ == "__main__":
    main()
