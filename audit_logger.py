#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Logger - Immutable Audit Trail System
Records all review actions with cryptographic integrity verification
"""

import json
import csv
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from persistence_utils import PersistenceManager, SchemaMigrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Single immutable audit log entry"""
    audit_id: str
    timestamp: str  # ISO format
    action: str  # REVIEW_COMPLETED, QUEUE_ACTION, FEEDBACK_PROCESSED, etc.
    reviewer_id: Optional[str]
    review_id: Optional[str]
    original_prediction: Optional[Dict[str, Any]]
    human_decision: Optional[Dict[str, Any]]
    additional_data: Optional[Dict[str, Any]]
    previous_hash: str
    entry_hash: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEntry':
        """Create AuditEntry from dictionary"""
        return cls(**data)


@dataclass
class ComplianceReport:
    """Compliance report with review coverage and accuracy metrics"""
    report_id: str
    generated_at: str
    start_date: str
    end_date: str
    total_reviews: int
    review_coverage: Dict[str, int]  # By check type
    accuracy_metrics: Dict[str, Any]
    reviewer_stats: Dict[str, Any]
    audit_integrity: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class AuditLogger:
    """
    Maintains immutable audit trail for compliance
    
    Features:
    - Immutable record storage with cryptographic hashing
    - Chain of custody verification
    - JSON and CSV export
    - Compliance report generation
    - Thread-safe operations
    - Tamper detection
    """
    
    def __init__(self, audit_dir: str = "./audit_logs/", 
                 current_log_file: str = "audit_log.json",
                 max_entries_per_file: int = 10000):
        """
        Initialize audit logger
        
        Args:
            audit_dir: Directory for audit log storage
            current_log_file: Name of current audit log file
            max_entries_per_file: Maximum entries before rotating log file
        """
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log_file = self.audit_dir / current_log_file
        self.max_entries_per_file = max_entries_per_file
        
        # In-memory audit entries
        self.audit_entries: List[AuditEntry] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing audit log
        self._load_audit_log()
        
        # Get last hash for chain
        self.last_hash = self._get_last_hash()
        
        logger.info(f"AuditLogger initialized with {len(self.audit_entries)} existing entries")
    
    def log_review(self, review_item, decision):
        """
        Log a completed review action
        
        Args:
            review_item: ReviewItem that was reviewed
            decision: ReviewDecision from reviewer
        """
        original_prediction = {
            'violation': review_item.predicted_violation,
            'confidence': review_item.confidence,
            'check_type': review_item.check_type,
            'ai_reasoning': review_item.ai_reasoning,
            'evidence': review_item.evidence,
            'severity': review_item.severity
        }
        
        human_decision = {
            'decision': decision.decision,
            'actual_violation': decision.actual_violation,
            'corrected_confidence': decision.corrected_confidence,
            'reviewer_notes': decision.reviewer_notes,
            'tags': decision.tags,
            'review_duration_seconds': decision.review_duration_seconds
        }
        
        additional_data = {
            'document_id': review_item.document_id,
            'slide': review_item.slide,
            'location': review_item.location,
            'rule': review_item.rule
        }
        
        self._create_audit_entry(
            action="REVIEW_COMPLETED",
            reviewer_id=decision.reviewer_id,
            review_id=review_item.review_id,
            original_prediction=original_prediction,
            human_decision=human_decision,
            additional_data=additional_data
        )
        
        logger.info(f"Logged review: {review_item.review_id} by {decision.reviewer_id}")

    def log_queue_action(self, action: str, details: Dict):
        """
        Log a queue management action
        
        Args:
            action: Action type (ITEM_ADDED, ITEM_ASSIGNED, BATCH_PROCESSED, etc.)
            details: Dictionary with action details
        """
        self._create_audit_entry(
            action=action,
            reviewer_id=details.get('reviewer_id'),
            review_id=details.get('review_id'),
            original_prediction=None,
            human_decision=None,
            additional_data=details
        )
        
        logger.debug(f"Logged queue action: {action}")
    
    def log_feedback_processing(self, feedback_record, processing_result):
        """
        Log feedback processing for learning system
        
        Args:
            feedback_record: FeedbackRecord that was processed
            processing_result: FeedbackProcessingResult
        """
        additional_data = {
            'feedback_id': feedback_record.feedback_id,
            'check_type': feedback_record.check_type,
            'processing_time_ms': processing_result.processing_time_ms,
            'calibration_updated': processing_result.calibration_updated,
            'patterns_analyzed': processing_result.patterns_analyzed,
            'confidence_adjusted': processing_result.confidence_adjusted,
            'error': processing_result.error
        }
        
        self._create_audit_entry(
            action="FEEDBACK_PROCESSED",
            reviewer_id=feedback_record.reviewer_id,
            review_id=None,
            original_prediction=None,
            human_decision=None,
            additional_data=additional_data
        )
        
        logger.debug(f"Logged feedback processing: {feedback_record.feedback_id}")
    
    def _create_audit_entry(self, action: str, reviewer_id: Optional[str],
                           review_id: Optional[str], original_prediction: Optional[Dict],
                           human_decision: Optional[Dict], additional_data: Optional[Dict]):
        """
        Create and store an immutable audit entry
        
        Args:
            action: Action type
            reviewer_id: ID of reviewer (if applicable)
            review_id: ID of review item (if applicable)
            original_prediction: Original AI prediction data
            human_decision: Human decision data
            additional_data: Additional context data
        """
        with self.lock:
            # Generate audit ID
            audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Create entry (without hash first)
            entry = AuditEntry(
                audit_id=audit_id,
                timestamp=datetime.now().isoformat(),
                action=action,
                reviewer_id=reviewer_id,
                review_id=review_id,
                original_prediction=original_prediction,
                human_decision=human_decision,
                additional_data=additional_data,
                previous_hash=self.last_hash,
                entry_hash=""  # Will be calculated
            )
            
            # Calculate hash for this entry
            entry.entry_hash = self._calculate_entry_hash(entry)
            
            # Add to entries
            self.audit_entries.append(entry)
            
            # Update last hash for chain
            self.last_hash = entry.entry_hash
            
            # Save to disk
            self._save_audit_log()
    
    def _calculate_entry_hash(self, entry: AuditEntry) -> str:
        """
        Calculate cryptographic hash for audit entry
        
        Args:
            entry: AuditEntry to hash
            
        Returns:
            SHA256 hash string
        """
        # Create deterministic string representation
        hash_data = {
            'audit_id': entry.audit_id,
            'timestamp': entry.timestamp,
            'action': entry.action,
            'reviewer_id': entry.reviewer_id,
            'review_id': entry.review_id,
            'original_prediction': entry.original_prediction,
            'human_decision': entry.human_decision,
            'additional_data': entry.additional_data,
            'previous_hash': entry.previous_hash
        }
        
        # Convert to JSON string (sorted keys for determinism)
        hash_string = json.dumps(hash_data, sort_keys=True)
        
        # Calculate SHA256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def verify_integrity(self, include_archived: bool = False) -> Dict[str, Any]:
        """
        Verify integrity of audit log chain
        
        Args:
            include_archived: Whether to include archived logs in verification
        
        Returns:
            Dict with verification results
        """
        with self.lock:
            if not self.audit_entries:
                return {
                    'valid': True,
                    'total_entries': 0,
                    'errors': [],
                    'note': 'No entries to verify'
                }
            
            errors = []
            
            # For the first entry, check if it's continuing from a previous log
            # (after rotation) or starting fresh
            first_entry = self.audit_entries[0]
            if first_entry.previous_hash == "genesis":
                # Fresh start
                previous_hash = "genesis"
            else:
                # Continuing from previous log (after rotation)
                # We can't verify the previous hash without the archived log
                previous_hash = first_entry.previous_hash
                if not include_archived:
                    # Note that we're skipping verification of the first link
                    errors.append({
                        'entry_index': 0,
                        'audit_id': first_entry.audit_id,
                        'error': 'Chain continues from archived log',
                        'note': 'Cannot verify without archived logs',
                        'severity': 'info'
                    })
            
            for i, entry in enumerate(self.audit_entries):
                # Skip first entry previous hash check if continuing from archive
                if i == 0 and entry.previous_hash != "genesis" and not include_archived:
                    previous_hash = entry.entry_hash
                    continue
                
                # Verify previous hash matches
                if entry.previous_hash != previous_hash:
                    errors.append({
                        'entry_index': i,
                        'audit_id': entry.audit_id,
                        'error': 'Previous hash mismatch',
                        'expected': previous_hash,
                        'actual': entry.previous_hash,
                        'severity': 'error'
                    })
                
                # Verify entry hash is correct
                calculated_hash = self._calculate_entry_hash(entry)
                if entry.entry_hash != calculated_hash:
                    errors.append({
                        'entry_index': i,
                        'audit_id': entry.audit_id,
                        'error': 'Entry hash mismatch',
                        'expected': calculated_hash,
                        'actual': entry.entry_hash,
                        'severity': 'error'
                    })
                
                previous_hash = entry.entry_hash
            
            # Filter out info-level messages for validity check
            critical_errors = [e for e in errors if e.get('severity') == 'error']
            
            result = {
                'valid': len(critical_errors) == 0,
                'total_entries': len(self.audit_entries),
                'errors': errors,
                'critical_errors': len(critical_errors),
                'info_messages': len(errors) - len(critical_errors)
            }
            
            if result['valid']:
                logger.info(f"Audit log integrity verified: {result['total_entries']} entries")
            else:
                logger.error(f"Audit log integrity check failed: {len(critical_errors)} critical errors found")
            
            return result
    
    def get_audit_trail(self, filters: Optional[Dict] = None) -> List[AuditEntry]:
        """
        Get audit trail with optional filtering
        
        Args:
            filters: Optional dict with filter criteria:
                - action: str
                - reviewer_id: str
                - review_id: str
                - start_date: str (ISO format)
                - end_date: str (ISO format)
                - check_type: str
                
        Returns:
            List of AuditEntry objects
        """
        with self.lock:
            entries = list(self.audit_entries)
            
            if not filters:
                return entries
            
            # Apply filters
            if 'action' in filters:
                entries = [e for e in entries if e.action == filters['action']]
            
            if 'reviewer_id' in filters:
                entries = [e for e in entries if e.reviewer_id == filters['reviewer_id']]
            
            if 'review_id' in filters:
                entries = [e for e in entries if e.review_id == filters['review_id']]
            
            if 'start_date' in filters:
                start = datetime.fromisoformat(filters['start_date'])
                entries = [e for e in entries 
                          if datetime.fromisoformat(e.timestamp) >= start]
            
            if 'end_date' in filters:
                end = datetime.fromisoformat(filters['end_date'])
                entries = [e for e in entries 
                          if datetime.fromisoformat(e.timestamp) <= end]
            
            if 'check_type' in filters:
                entries = [e for e in entries 
                          if e.original_prediction and 
                          e.original_prediction.get('check_type') == filters['check_type']]
            
            return entries
    
    def export_audit_log(self, filepath: str, format: str = 'json',
                        filters: Optional[Dict] = None):
        """
        Export audit log to file
        
        Args:
            filepath: Output file path
            format: Export format ('json' or 'csv')
            filters: Optional filters to apply
        """
        entries = self.get_audit_trail(filters)
        
        if format.lower() == 'json':
            self._export_json(filepath, entries)
        elif format.lower() == 'csv':
            self._export_csv(filepath, entries)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(entries)} audit entries to {filepath} ({format})")
    
    def _export_json(self, filepath: str, entries: List[AuditEntry]):
        """Export audit entries to JSON"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_entries': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, filepath: str, entries: List[AuditEntry]):
        """Export audit entries to CSV"""
        if not entries:
            # Create empty CSV with headers
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'audit_id', 'timestamp', 'action', 'reviewer_id', 'review_id',
                    'predicted_violation', 'predicted_confidence', 'check_type',
                    'actual_violation', 'decision', 'reviewer_notes',
                    'entry_hash'
                ])
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'audit_id', 'timestamp', 'action', 'reviewer_id', 'review_id',
                'predicted_violation', 'predicted_confidence', 'check_type',
                'actual_violation', 'decision', 'reviewer_notes',
                'entry_hash'
            ])
            
            # Write entries
            for entry in entries:
                writer.writerow([
                    entry.audit_id,
                    entry.timestamp,
                    entry.action,
                    entry.reviewer_id or '',
                    entry.review_id or '',
                    entry.original_prediction.get('violation', '') if entry.original_prediction else '',
                    entry.original_prediction.get('confidence', '') if entry.original_prediction else '',
                    entry.original_prediction.get('check_type', '') if entry.original_prediction else '',
                    entry.human_decision.get('actual_violation', '') if entry.human_decision else '',
                    entry.human_decision.get('decision', '') if entry.human_decision else '',
                    entry.human_decision.get('reviewer_notes', '') if entry.human_decision else '',
                    entry.entry_hash
                ])

    def generate_compliance_report(self, start_date: str, end_date: str) -> ComplianceReport:
        """
        Generate compliance report for date range
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            ComplianceReport object
        """
        # Get entries in date range
        filters = {
            'start_date': start_date,
            'end_date': end_date
        }
        entries = self.get_audit_trail(filters)
        
        # Filter to review completed actions
        review_entries = [e for e in entries if e.action == 'REVIEW_COMPLETED']
        
        # Calculate review coverage by check type
        review_coverage = {}
        for entry in review_entries:
            if entry.original_prediction:
                check_type = entry.original_prediction.get('check_type', 'UNKNOWN')
                review_coverage[check_type] = review_coverage.get(check_type, 0) + 1
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(review_entries)
        
        # Calculate reviewer statistics
        reviewer_stats = self._calculate_reviewer_stats(review_entries)
        
        # Verify audit integrity
        integrity_check = self.verify_integrity()
        audit_integrity = {
            'valid': integrity_check['valid'],
            'total_entries': integrity_check['total_entries'],
            'errors_found': len(integrity_check['errors'])
        }
        
        # Create report
        report = ComplianceReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            start_date=start_date,
            end_date=end_date,
            total_reviews=len(review_entries),
            review_coverage=review_coverage,
            accuracy_metrics=accuracy_metrics,
            reviewer_stats=reviewer_stats,
            audit_integrity=audit_integrity
        )
        
        logger.info(f"Generated compliance report: {report.report_id}")
        
        return report
    
    def _calculate_accuracy_metrics(self, review_entries: List[AuditEntry]) -> Dict[str, Any]:
        """Calculate accuracy metrics from review entries"""
        if not review_entries:
            return {
                'total_reviews': 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0
            }
        
        # Calculate confusion matrix
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for entry in review_entries:
            if not entry.original_prediction or not entry.human_decision:
                continue
            
            predicted = entry.original_prediction.get('violation', False)
            actual = entry.human_decision.get('actual_violation', False)
            
            if predicted and actual:
                true_positives += 1
            elif predicted and not actual:
                false_positives += 1
            elif not predicted and not actual:
                true_negatives += 1
            elif not predicted and actual:
                false_negatives += 1
        
        # Calculate metrics
        total = true_positives + false_positives + true_negatives + false_negatives
        
        precision = (true_positives / (true_positives + false_positives) 
                    if (true_positives + false_positives) > 0 else 0.0)
        recall = (true_positives / (true_positives + false_negatives) 
                 if (true_positives + false_negatives) > 0 else 0.0)
        f1_score = (2 * (precision * recall) / (precision + recall) 
                   if (precision + recall) > 0 else 0.0)
        accuracy = ((true_positives + true_negatives) / total 
                   if total > 0 else 0.0)
        
        return {
            'total_reviews': len(review_entries),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
    
    def _calculate_reviewer_stats(self, review_entries: List[AuditEntry]) -> Dict[str, Any]:
        """Calculate reviewer statistics"""
        reviewer_data = {}
        
        for entry in review_entries:
            if not entry.reviewer_id:
                continue
            
            if entry.reviewer_id not in reviewer_data:
                reviewer_data[entry.reviewer_id] = {
                    'total_reviews': 0,
                    'total_duration_seconds': 0,
                    'decisions': {'APPROVE': 0, 'REJECT': 0, 'MODIFY': 0}
                }
            
            reviewer_data[entry.reviewer_id]['total_reviews'] += 1
            
            if entry.human_decision:
                duration = entry.human_decision.get('review_duration_seconds', 0)
                reviewer_data[entry.reviewer_id]['total_duration_seconds'] += duration
                
                decision = entry.human_decision.get('decision', 'UNKNOWN')
                if decision in reviewer_data[entry.reviewer_id]['decisions']:
                    reviewer_data[entry.reviewer_id]['decisions'][decision] += 1
        
        # Calculate averages
        for reviewer_id, data in reviewer_data.items():
            if data['total_reviews'] > 0:
                data['avg_review_time_seconds'] = (
                    data['total_duration_seconds'] / data['total_reviews']
                )
        
        return reviewer_data
    
    def export_compliance_report(self, report: ComplianceReport, filepath: str):
        """
        Export compliance report to JSON file
        
        Args:
            report: ComplianceReport to export
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Exported compliance report to {filepath}")
    
    def print_compliance_report(self, report: ComplianceReport):
        """Print compliance report to console"""
        print("\n" + "="*70)
        print("Compliance Report")
        print("="*70)
        print(f"\nReport ID: {report.report_id}")
        print(f"Generated: {report.generated_at}")
        print(f"Period: {report.start_date} to {report.end_date}")
        
        print(f"\n📊 Review Summary:")
        print(f"  Total Reviews: {report.total_reviews}")
        
        print(f"\n📋 Review Coverage by Check Type:")
        for check_type, count in sorted(report.review_coverage.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  {check_type}: {count}")
        
        print(f"\n🎯 Accuracy Metrics:")
        metrics = report.accuracy_metrics
        print(f"  Accuracy: {metrics['accuracy']:.1%}")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall: {metrics['recall']:.1%}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives: {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        
        print(f"\n👥 Reviewer Statistics:")
        for reviewer_id, stats in report.reviewer_stats.items():
            print(f"  {reviewer_id}:")
            print(f"    Total Reviews: {stats['total_reviews']}")
            print(f"    Avg Review Time: {stats.get('avg_review_time_seconds', 0):.1f}s")
            print(f"    Decisions: {stats['decisions']}")
        
        print(f"\n🔒 Audit Integrity:")
        integrity = report.audit_integrity
        print(f"  Valid: {'✓ Yes' if integrity['valid'] else '✗ No'}")
        print(f"  Total Entries: {integrity['total_entries']}")
        print(f"  Errors Found: {integrity['errors_found']}")
        
        print("\n" + "="*70 + "\n")
    
    def _get_last_hash(self) -> str:
        """Get hash of last entry in chain"""
        if not self.audit_entries:
            return "genesis"
        return self.audit_entries[-1].entry_hash
    
    def _load_audit_log(self):
        """Load audit log from file with thread-safe file locking"""
        try:
            if not self.current_log_file.exists():
                logger.info("No existing audit log found, starting fresh")
                return
            
            # Setup schema migrator
            migrator = SchemaMigrator(current_version=2)
            migrator.register_migration(1, 2, self._migrate_audit_v1_to_v2)
            
            # Load data with migration support
            data = PersistenceManager.load_json(
                str(self.current_log_file),
                migration_func=migrator.migrate
            )
            
            if not data:
                return
            
            # Load entries
            for entry_data in data.get('entries', []):
                entry = AuditEntry.from_dict(entry_data)
                self.audit_entries.append(entry)
            
            schema_version = data.get('schema_version', 1)
            logger.info(f"Loaded {len(self.audit_entries)} audit entries from {self.current_log_file} (schema v{schema_version})")
            
        except Exception as e:
            logger.error(f"Error loading audit log: {e}")
    
    def _save_audit_log(self):
        """Save audit log to file with thread-safe file locking and rotation"""
        try:
            # Check if rotation is needed
            if len(self.audit_entries) >= self.max_entries_per_file:
                self._rotate_audit_log()
            
            # Prepare data
            data = {
                'schema_version': 2,  # Current schema version
                'last_updated': datetime.now().isoformat(),
                'total_entries': len(self.audit_entries),
                'entries': [entry.to_dict() for entry in self.audit_entries]
            }
            
            # Save using persistence manager
            PersistenceManager.save_json(str(self.current_log_file), data, backup=True)
            
            logger.debug(f"Saved audit log to {self.current_log_file}")
            
        except Exception as e:
            logger.error(f"Error saving audit log: {e}")
    
    def _rotate_audit_log(self):
        """
        Rotate audit log when it reaches max size
        Archives current log and starts fresh
        """
        try:
            # Generate archive filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"audit_log_{timestamp}.json"
            archive_path = self.audit_dir / archive_name
            
            # Save current entries to archive
            archive_data = {
                'schema_version': 2,
                'archived_at': datetime.now().isoformat(),
                'total_entries': len(self.audit_entries),
                'entries': [entry.to_dict() for entry in self.audit_entries]
            }
            
            with open(archive_path, 'w') as f:
                json.dump(archive_data, f, indent=2)
            
            logger.info(f"Rotated audit log: archived {len(self.audit_entries)} entries to {archive_name}")
            
            # Clear in-memory entries but keep last hash for chain continuity
            # This maintains the cryptographic chain across rotations
            if self.audit_entries:
                self.last_hash = self.audit_entries[-1].entry_hash
            self.audit_entries = []
            
        except Exception as e:
            logger.error(f"Error rotating audit log: {e}")
    
    def _migrate_audit_v1_to_v2(self, data: Dict) -> Dict:
        """
        Migrate audit data from v1 to v2
        
        Args:
            data: Audit data in v1 schema
            
        Returns:
            Migrated data in v2 schema
        """
        logger.info("Migrating audit schema from v1 to v2")
        
        # v1 didn't have schema_version field
        # v2 adds schema_version and ensures all entries have required fields
        
        entries = data.get('entries', [])
        for entry in entries:
            # Add any missing fields with defaults
            if 'additional_data' not in entry:
                entry['additional_data'] = None
            if 'previous_hash' not in entry:
                entry['previous_hash'] = 'genesis'
            if 'entry_hash' not in entry:
                # Recalculate hash if missing
                temp_entry = AuditEntry.from_dict(entry)
                entry['entry_hash'] = self._calculate_entry_hash(temp_entry)
        
        data['schema_version'] = 2
        
        return data
    
    def load_archived_logs(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> List[AuditEntry]:
        """
        Load entries from archived audit logs
        
        Args:
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            List of AuditEntry objects from archives
        """
        archived_entries = []
        
        try:
            # Find all archived log files
            archive_files = sorted(self.audit_dir.glob("audit_log_*.json"))
            
            for archive_file in archive_files:
                try:
                    with open(archive_file, 'r') as f:
                        data = json.load(f)
                    
                    # Load entries from archive
                    for entry_data in data.get('entries', []):
                        entry = AuditEntry.from_dict(entry_data)
                        
                        # Apply date filters if provided
                        if start_date:
                            if entry.timestamp < start_date:
                                continue
                        if end_date:
                            if entry.timestamp > end_date:
                                continue
                        
                        archived_entries.append(entry)
                    
                except Exception as e:
                    logger.error(f"Error loading archive {archive_file}: {e}")
            
            logger.info(f"Loaded {len(archived_entries)} entries from {len(archive_files)} archived logs")
            
        except Exception as e:
            logger.error(f"Error loading archived logs: {e}")
        
        return archived_entries


if __name__ == "__main__":
    # Example usage and testing
    print("="*70)
    print("Audit Logger - Immutable Audit Trail System")
    print("="*70)
    
    # Initialize audit logger
    audit_logger = AuditLogger(audit_dir="./test_audit_logs/")
    
    print(f"\n✓ Audit logger initialized")
    print(f"  Audit directory: ./test_audit_logs/")
    print(f"  Existing entries: {len(audit_logger.audit_entries)}")
    
    # Simulate some audit entries
    print("\n📝 Creating test audit entries...")
    
    # Mock review item and decision
    from review_manager import ReviewItem, ReviewDecision, ReviewStatus
    
    for i in range(3):
        review_item = ReviewItem(
            review_id=f"review_{i}",
            document_id=f"doc_{i}",
            check_type="PROMOTIONAL_MENTION",
            slide=f"Slide {i+1}",
            location="Header",
            predicted_violation=True,
            confidence=65 + (i * 5),
            ai_reasoning="AI detected promotional mention",
            evidence=f"Found phrase on slide {i+1}",
            rule="Promotional mention required",
            severity="HIGH",
            created_at=datetime.now().isoformat(),
            priority_score=50.0,
            status=ReviewStatus.REVIEWED
        )
        
        decision = ReviewDecision(
            review_id=f"review_{i}",
            reviewer_id="reviewer_001",
            decision="APPROVE" if i % 2 == 0 else "REJECT",
            actual_violation=True if i % 2 == 0 else False,
            corrected_confidence=None,
            reviewer_notes="Confirmed" if i % 2 == 0 else "False positive",
            tags=["reviewed"],
            reviewed_at=datetime.now().isoformat(),
            review_duration_seconds=120
        )
        
        audit_logger.log_review(review_item, decision)
    
    print(f"  ✓ Created {3} audit entries")
    
    # Verify integrity
    print("\n🔒 Verifying audit log integrity...")
    integrity = audit_logger.verify_integrity()
    print(f"  Valid: {'✓ Yes' if integrity['valid'] else '✗ No'}")
    print(f"  Total Entries: {integrity['total_entries']}")
    print(f"  Errors: {len(integrity['errors'])}")
    
    # Export to JSON
    print("\n💾 Exporting audit log...")
    audit_logger.export_audit_log("test_audit_export.json", format="json")
    print(f"  ✓ Exported to test_audit_export.json")
    
    # Export to CSV
    audit_logger.export_audit_log("test_audit_export.csv", format="csv")
    print(f"  ✓ Exported to test_audit_export.csv")
    
    # Generate compliance report
    print("\n📊 Generating compliance report...")
    start_date = (datetime.now() - timedelta(days=30)).isoformat()
    end_date = datetime.now().isoformat()
    
    report = audit_logger.generate_compliance_report(start_date, end_date)
    audit_logger.print_compliance_report(report)
    
    # Export report
    audit_logger.export_compliance_report(report, "test_compliance_report.json")
    print(f"✓ Exported compliance report to test_compliance_report.json")
    
    print("\n" + "="*70)
    print("✓ Audit Logger test complete")
    print("="*70)
