#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script showing batch review operations
"""

import os
from datetime import datetime
from review_manager import ReviewManager, ReviewItem, ReviewStatus

def create_demo_queue():
    """Create a demo review queue with various items"""
    
    # Clean up any existing demo queue
    demo_queue_file = "demo_review_queue.json"
    if os.path.exists(demo_queue_file):
        os.remove(demo_queue_file)
    
    # Initialize manager
    manager = ReviewManager(queue_file=demo_queue_file)
    
    print("="*70)
    print("Creating Demo Review Queue for Batch Operations")
    print("="*70)
    print()
    
    # Create diverse test items
    demo_items = [
        # Test document items (should be rejected)
        ("test_001", "test_doc_001", "STRUCTURE", "LOW", 65, "Test promotional mention"),
        ("test_002", "test_doc_001", "PERFORMANCE", "LOW", 68, "Test performance data"),
        ("test_003", "test_doc_001", "ESG", "LOW", 62, "Test ESG disclosure"),
        
        # Real document with STRUCTURE issues
        ("real_001", "prospectus_2024", "STRUCTURE", "CRITICAL", 58, "Missing required section"),
        ("real_002", "prospectus_2024", "STRUCTURE", "HIGH", 61, "Incorrect heading format"),
        ("real_003", "prospectus_2024", "STRUCTURE", "MEDIUM", 64, "Section order issue"),
        
        # Real document with PERFORMANCE issues
        ("real_004", "fund_report_q4", "PERFORMANCE", "HIGH", 55, "Incomplete performance table"),
        ("real_005", "fund_report_q4", "PERFORMANCE", "MEDIUM", 59, "Missing benchmark comparison"),
        
        # Real document with ESG issues
        ("real_006", "esg_report_2024", "ESG", "CRITICAL", 52, "Missing ESG disclosure"),
        ("real_007", "esg_report_2024", "ESG", "HIGH", 57, "Incomplete sustainability metrics"),
        
        # Real document with VALUES issues
        ("real_008", "marketing_doc", "VALUES", "MEDIUM", 70, "Potential promotional language"),
        ("real_009", "marketing_doc", "VALUES", "LOW", 72, "Unclear value statement"),
    ]
    
    print("Creating review items:\n")
    
    for review_id, doc_id, check_type, severity, confidence, reasoning in demo_items:
        item = ReviewItem(
            review_id=review_id,
            document_id=doc_id,
            check_type=check_type,
            slide=f"Slide {review_id.split('_')[1]}",
            location="Main content section",
            predicted_violation=True,
            confidence=confidence,
            ai_reasoning=reasoning,
            evidence=f"Evidence for {review_id}",
            rule=f"{check_type} compliance rule",
            severity=severity,
            created_at=datetime.now().isoformat(),
            priority_score=manager.calculate_priority_score(confidence, severity),
            status=ReviewStatus.PENDING
        )
        manager.add_to_queue(item)
        
        doc_type = "TEST" if doc_id.startswith("test_") else "REAL"
        print(f"  [{doc_type}] {review_id}: {check_type} - {severity} - {confidence}%")
    
    print(f"\n✓ Created {len(demo_items)} review items")
    
    # Show statistics
    stats = manager.get_queue_stats()
    print(f"\nQueue Statistics:")
    print(f"  Total Pending: {stats.total_pending}")
    print(f"  By Check Type: {stats.by_check_type}")
    print(f"  By Severity: {stats.by_severity}")
    print(f"  Avg Confidence: {stats.avg_confidence:.1f}%")
    
    print("\n" + "="*70)
    print("Demo Queue Created Successfully!")
    print("="*70)
    print("\nYou can now use the review CLI with batch operations:")
    print("\nExample commands:")
    print("  # Start interactive review")
    print("  python review.py --queue-file=demo_review_queue.json")
    print()
    print("  # Batch reject all test document items")
    print("  python review.py batch --queue-file=demo_review_queue.json \\")
    print("    --document=test_doc_001 --action=reject --notes 'Test data'")
    print()
    print("  # Batch approve all LOW severity items")
    print("  python review.py batch --queue-file=demo_review_queue.json \\")
    print("    --severity=LOW --action=approve")
    print()
    print("  # Batch review STRUCTURE checks")
    print("  python review.py batch --queue-file=demo_review_queue.json \\")
    print("    --check-type=STRUCTURE --action=approve --limit=3")
    print()
    print("  # View queue status")
    print("  python review.py status --queue-file=demo_review_queue.json")
    print()

if __name__ == "__main__":
    create_demo_queue()
