#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review CLI - Interactive Command-Line Interface for Human-in-the-Loop Reviews
Provides commands for reviewing flagged compliance violations
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from review_manager import ReviewManager, ReviewItem, ReviewDecision, ReviewStatus
from review_metrics import MetricsTracker
from audit_logger import AuditLogger

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def disable():
        """Disable colors for non-terminal output"""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''
        Colors.END = ''


class ReviewCLI:
    """
    Command-line interface for interactive review sessions
    
    Features:
    - Display pending reviews with full context
    - Approve/reject/skip reviews
    - Show queue status and progress
    - Batch operations
    - Progress indicators
    """
    
    def __init__(self, reviewer_id: str = "default_reviewer", queue_file: str = "review_queue.json",
                 audit_dir: str = "./audit_logs/"):
        """
        Initialize review CLI
        
        Args:
            reviewer_id: ID of the reviewer
            queue_file: Path to review queue file
            audit_dir: Directory for audit logs
        """
        self.reviewer_id = reviewer_id
        
        # Initialize audit logger
        self.audit_logger = AuditLogger(audit_dir=audit_dir)
        
        # Initialize review manager with audit logger
        self.manager = ReviewManager(queue_file=queue_file, audit_logger=self.audit_logger)
        self.metrics_tracker = MetricsTracker(self.manager)
        self.current_item: Optional[ReviewItem] = None
        self.review_start_time: Optional[datetime] = None
        self.session_stats = {
            'reviewed': 0,
            'approved': 0,
            'rejected': 0,
            'skipped': 0
        }
        
        # Disable colors if not in terminal
        if not sys.stdout.isatty():
            Colors.disable()
    
    def start_interactive_session(self):
        """Start an interactive review session"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🔍 Human-in-the-Loop Review Session{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"Reviewer: {Colors.BOLD}{self.reviewer_id}{Colors.END}\n")
        
        # Show initial status
        self.show_status()
        
        print(f"\n{Colors.BOLD}Commands:{Colors.END}")
        print(f"  {Colors.GREEN}next{Colors.END}       - View next pending review")
        print(f"  {Colors.GREEN}approve{Colors.END}    - Approve current review (optional: --notes 'text')")
        print(f"  {Colors.GREEN}reject{Colors.END}     - Reject current review (required: --notes 'text')")
        print(f"  {Colors.GREEN}skip{Colors.END}       - Skip current review")
        print(f"  {Colors.GREEN}status{Colors.END}     - Show queue status")
        print(f"  {Colors.GREEN}metrics{Colors.END}    - Show review metrics and performance statistics")
        print(f"  {Colors.GREEN}batch{Colors.END}      - Batch review operations (see 'help batch')")
        print(f"  {Colors.GREEN}export{Colors.END}     - Export audit log (see 'help export')")
        print(f"  {Colors.GREEN}report{Colors.END}     - Generate compliance report (see 'help report')")
        print(f"  {Colors.GREEN}quit{Colors.END}       - Exit review session")
        
        print(f"\n{Colors.YELLOW}Type 'next' to begin reviewing{Colors.END}\n")
        
        # Interactive loop
        while True:
            try:
                command = input(f"{Colors.BOLD}review> {Colors.END}").strip()
                
                if not command:
                    continue
                
                # Parse command
                parts = command.split()
                cmd = parts[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    self._show_session_summary()
                    break
                elif cmd == 'next':
                    self.cmd_next()
                elif cmd == 'approve':
                    notes = self._extract_notes(command)
                    self.cmd_approve(notes)
                elif cmd == 'reject':
                    notes = self._extract_notes(command)
                    self.cmd_reject(notes)
                elif cmd == 'skip':
                    self.cmd_skip()
                elif cmd == 'status':
                    self.show_status()
                elif cmd == 'metrics':
                    self.cmd_metrics(command)
                elif cmd == 'batch':
                    self.cmd_batch(command)
                elif cmd == 'export':
                    self.cmd_export(command)
                elif cmd == 'report':
                    self.cmd_report(command)
                elif cmd == 'help':
                    if len(parts) > 1 and parts[1] == 'batch':
                        self._show_batch_help()
                    else:
                        self._show_help()
                else:
                    print(f"{Colors.RED}Unknown command: {cmd}{Colors.END}")
                    print(f"Type 'help' for available commands\n")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Session interrupted{Colors.END}")
                self._show_session_summary()
                break
            except EOFError:
                print(f"\n\n{Colors.YELLOW}Session ended{Colors.END}")
                self._show_session_summary()
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.END}\n")
                logger.exception("Error in interactive session")
    
    def cmd_next(self):
        """Display next pending review"""
        # Get next item from queue
        self.current_item = self.manager.get_next_review(self.reviewer_id)
        
        if not self.current_item:
            print(f"\n{Colors.GREEN}✓ No more reviews pending!{Colors.END}\n")
            return
        
        # Start timing
        self.review_start_time = datetime.now()
        
        # Display review item
        self.display_review_item(self.current_item)
        
        # Show progress
        stats = self.manager.get_queue_stats()
        total = stats.total_pending + stats.total_in_review + stats.total_reviewed
        completed = stats.total_reviewed
        self.show_progress(completed, total)
    
    def cmd_approve(self, notes: str = ""):
        """
        Approve current review
        
        Args:
            notes: Optional reviewer notes
        """
        if not self.current_item:
            print(f"{Colors.RED}No review in progress. Use 'next' to load a review.{Colors.END}\n")
            return
        
        # Calculate review duration
        duration = 0
        if self.review_start_time:
            duration = int((datetime.now() - self.review_start_time).total_seconds())
        
        # Create decision
        decision = ReviewDecision(
            review_id=self.current_item.review_id,
            reviewer_id=self.reviewer_id,
            decision="APPROVE",
            actual_violation=True,  # Approved means violation is confirmed
            corrected_confidence=None,
            reviewer_notes=notes if notes else "Approved",
            tags=["approved"],
            reviewed_at=datetime.now().isoformat(),
            review_duration_seconds=duration
        )
        
        # Mark as reviewed
        success = self.manager.mark_reviewed(self.current_item.review_id, decision)
        
        if success:
            print(f"\n{Colors.GREEN}✓ Review approved{Colors.END}")
            if notes:
                print(f"  Notes: {notes}")
            print()
            
            # Update session stats
            self.session_stats['reviewed'] += 1
            self.session_stats['approved'] += 1
            
            # Clear current item
            self.current_item = None
            self.review_start_time = None
        else:
            print(f"{Colors.RED}Failed to mark review as approved{Colors.END}\n")
    
    def cmd_reject(self, notes: str = ""):
        """
        Reject current review
        
        Args:
            notes: Required explanatory notes
        """
        if not self.current_item:
            print(f"{Colors.RED}No review in progress. Use 'next' to load a review.{Colors.END}\n")
            return
        
        # Require notes for rejection
        if not notes:
            print(f"{Colors.RED}Rejection requires explanatory notes.{Colors.END}")
            print(f"Usage: reject --notes 'Explanation for rejection'\n")
            return
        
        # Calculate review duration
        duration = 0
        if self.review_start_time:
            duration = int((datetime.now() - self.review_start_time).total_seconds())
        
        # Create decision
        decision = ReviewDecision(
            review_id=self.current_item.review_id,
            reviewer_id=self.reviewer_id,
            decision="REJECT",
            actual_violation=False,  # Rejected means no violation
            corrected_confidence=None,
            reviewer_notes=notes,
            tags=["rejected", "false_positive"],
            reviewed_at=datetime.now().isoformat(),
            review_duration_seconds=duration
        )
        
        # Mark as reviewed
        success = self.manager.mark_reviewed(self.current_item.review_id, decision)
        
        if success:
            print(f"\n{Colors.YELLOW}✓ Review rejected{Colors.END}")
            print(f"  Notes: {notes}")
            print()
            
            # Update session stats
            self.session_stats['reviewed'] += 1
            self.session_stats['rejected'] += 1
            
            # Clear current item
            self.current_item = None
            self.review_start_time = None
        else:
            print(f"{Colors.RED}Failed to mark review as rejected{Colors.END}\n")
    
    def cmd_skip(self):
        """Skip current review and move to next"""
        if not self.current_item:
            print(f"{Colors.RED}No review in progress. Use 'next' to load a review.{Colors.END}\n")
            return
        
        # Return item to pending queue (by not marking as reviewed)
        # Just clear current item
        print(f"\n{Colors.YELLOW}⏭ Review skipped{Colors.END}\n")
        
        self.session_stats['skipped'] += 1
        self.current_item = None
        self.review_start_time = None
    
    def cmd_metrics(self, command: str = ""):
        """
        Display review metrics and performance statistics
        
        Args:
            command: Full command string with optional parameters
        """
        # Parse parameters for days filter
        days = 30  # Default
        
        if '--days' in command:
            parts = command.split('--days')
            if len(parts) > 1:
                try:
                    days_str = parts[1].strip().split()[0]
                    days = int(days_str)
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Invalid --days parameter. Using default (30).{Colors.END}\n")
        
        # Check for report generation flag
        if '--report' in command:
            print(f"\n{Colors.CYAN}Generating performance report...{Colors.END}\n")
            report = self.metrics_tracker.generate_performance_report(days=days)
            
            # Export report
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.metrics_tracker.export_report(report, filename)
            
            print(f"{Colors.GREEN}✓ Report generated and exported to {filename}{Colors.END}\n")
        else:
            # Display metrics summary
            self.metrics_tracker.print_metrics_summary(days=days)
    
    def cmd_batch(self, command: str):
        """
        Execute batch review operations
        
        Args:
            command: Full command string with parameters
        """
        # Parse batch command parameters
        params = self._parse_batch_params(command)
        
        if not params:
            print(f"{Colors.RED}Invalid batch command. Use 'help batch' for usage.{Colors.END}\n")
            return
        
        # Get batch items based on selection criteria
        batch_items = self._get_batch_items(params)
        
        if not batch_items:
            print(f"{Colors.YELLOW}No items found matching batch criteria.{Colors.END}\n")
            return
        
        # Display batch summary
        print(f"\n{Colors.BOLD}{'─'*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📦 Batch Review{Colors.END}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.END}\n")
        
        print(f"{Colors.BOLD}Selection Criteria:{Colors.END}")
        for key, value in params.items():
            if key not in ['action', 'notes', 'limit']:
                print(f"  {key}: {value}")
        
        print(f"\n{Colors.BOLD}Items Found:{Colors.END} {len(batch_items)}")
        
        # Show sample items
        print(f"\n{Colors.BOLD}Sample Items:{Colors.END}")
        for i, item in enumerate(batch_items[:3]):
            print(f"  {i+1}. {item.review_id} - {item.check_type} "
                  f"(confidence: {item.confidence}%, severity: {item.severity})")
        
        if len(batch_items) > 3:
            print(f"  ... and {len(batch_items) - 3} more")
        
        # Confirm action
        action = params.get('action', '').upper()
        if action not in ['APPROVE', 'REJECT']:
            print(f"\n{Colors.RED}Invalid action. Must be 'approve' or 'reject'.{Colors.END}\n")
            return
        
        # Require notes for rejection
        notes = params.get('notes', '')
        if action == 'REJECT' and not notes:
            print(f"\n{Colors.RED}Batch rejection requires --notes parameter.{Colors.END}\n")
            return
        
        # Confirm with user
        print(f"\n{Colors.YELLOW}Action:{Colors.END} {action} all {len(batch_items)} items")
        if notes:
            print(f"{Colors.YELLOW}Notes:{Colors.END} {notes}")
        
        confirm = input(f"\n{Colors.BOLD}Confirm batch operation? (yes/no): {Colors.END}").strip().lower()
        
        if confirm not in ['yes', 'y']:
            print(f"{Colors.YELLOW}Batch operation cancelled.{Colors.END}\n")
            return
        
        # Execute batch operation
        self._execute_batch_operation(batch_items, action, notes)
    
    def _parse_batch_params(self, command: str) -> Optional[Dict[str, str]]:
        """
        Parse batch command parameters
        
        Args:
            command: Full command string
            
        Returns:
            Dictionary of parameters or None if invalid
        """
        params = {}
        
        # Extract parameters
        parts = command.split('--')
        
        for part in parts[1:]:  # Skip first part (command name)
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                params[key] = value
            elif part.startswith('notes '):
                # Handle --notes without =
                notes = part[6:].strip().strip('"').strip("'")
                params['notes'] = notes
        
        return params if params else None
    
    def _get_batch_items(self, params: Dict[str, str]) -> List[ReviewItem]:
        """
        Get batch items based on selection criteria
        
        Args:
            params: Dictionary of selection parameters
            
        Returns:
            List of ReviewItem objects
        """
        limit = int(params.get('limit', 0)) or None
        
        # Check type selection
        if 'check-type' in params or 'check_type' in params:
            check_type = params.get('check-type') or params.get('check_type')
            return self.manager.get_batch_by_check_type(check_type, limit)
        
        # Document selection
        if 'document' in params:
            document_id = params['document']
            return self.manager.get_batch_by_document(document_id, limit)
        
        # Severity selection
        if 'severity' in params:
            severity = params['severity']
            return self.manager.get_batch_by_severity(severity, limit)
        
        # Confidence range selection
        if 'min-confidence' in params or 'max-confidence' in params:
            min_conf = int(params.get('min-confidence', 0))
            max_conf = int(params.get('max-confidence', 100))
            return self.manager.get_batch_by_confidence_range(min_conf, max_conf, limit)
        
        # Similarity selection (requires current item)
        if 'similar' in params and self.current_item:
            threshold = float(params.get('threshold', 0.85))
            return self.manager.get_similar_items(self.current_item.review_id, threshold)
        
        return []
    
    def _execute_batch_operation(self, batch_items: List[ReviewItem], 
                                 action: str, notes: str):
        """
        Execute batch review operation
        
        Args:
            batch_items: List of items to process
            action: Action to perform (APPROVE or REJECT)
            notes: Reviewer notes
        """
        print(f"\n{Colors.CYAN}Processing batch...{Colors.END}\n")
        
        processed = 0
        failed = 0
        
        # Process each item individually
        for item in batch_items:
            # Create individual decision for each item
            decision = ReviewDecision(
                review_id=item.review_id,
                reviewer_id=self.reviewer_id,
                decision=action,
                actual_violation=(action == 'APPROVE'),
                corrected_confidence=None,
                reviewer_notes=notes if notes else f"Batch {action.lower()}",
                tags=[f"batch_{action.lower()}"],
                reviewed_at=datetime.now().isoformat(),
                review_duration_seconds=0  # Batch operations don't track individual time
            )
            
            # Move item to in_review first (required by mark_reviewed)
            if item.review_id in self.manager.pending_items:
                with self.manager.lock:
                    item.status = ReviewStatus.IN_REVIEW
                    item.assigned_to = self.reviewer_id
                    self.manager.in_review_items[item.review_id] = item
                    del self.manager.pending_items[item.review_id]
            
            # Mark as reviewed
            if self.manager.mark_reviewed(item.review_id, decision):
                processed += 1
                print(f"  {Colors.GREEN}✓{Colors.END} {item.review_id}")
            else:
                failed += 1
                print(f"  {Colors.RED}✗{Colors.END} {item.review_id}")
        
        # Update session stats
        self.session_stats['reviewed'] += processed
        if action == 'APPROVE':
            self.session_stats['approved'] += processed
        else:
            self.session_stats['rejected'] += processed
        
        # Show summary
        print(f"\n{Colors.BOLD}Batch Operation Complete:{Colors.END}")
        print(f"  Processed: {Colors.GREEN}{processed}{Colors.END}")
        if failed > 0:
            print(f"  Failed:    {Colors.RED}{failed}{Colors.END}")
        print()
    
    def cmd_export(self, command: str = ""):
        """
        Export audit log to file
        
        Args:
            command: Full command string with parameters
        """
        # Parse parameters
        format_type = 'json'
        output_file = None
        
        if '--format=' in command:
            format_type = command.split('--format=')[1].split()[0].strip()
        
        if '--output=' in command:
            output_file = command.split('--output=')[1].split()[0].strip()
        
        # Generate default filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"audit_export_{timestamp}.{format_type}"
        
        # Export audit log
        print(f"\n{Colors.CYAN}Exporting audit log...{Colors.END}\n")
        
        try:
            self.audit_logger.export_audit_log(output_file, format=format_type)
            print(f"{Colors.GREEN}✓ Audit log exported to {output_file}{Colors.END}\n")
        except Exception as e:
            print(f"{Colors.RED}✗ Export failed: {e}{Colors.END}\n")
    
    def cmd_report(self, command: str = ""):
        """
        Generate compliance report
        
        Args:
            command: Full command string with parameters
        """
        # Parse parameters
        days = 30
        if '--days=' in command or '--days ' in command:
            try:
                days_str = command.split('--days')[1].split()[0].strip('=').strip()
                days = int(days_str)
            except:
                days = 30
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"\n{Colors.CYAN}Generating compliance report for last {days} days...{Colors.END}\n")
        
        try:
            # Generate report
            report = self.audit_logger.generate_compliance_report(
                start_date.isoformat(),
                end_date.isoformat()
            )
            
            # Display report
            self.audit_logger.print_compliance_report(report)
            
            # Export report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"compliance_report_{timestamp}.json"
            self.audit_logger.export_compliance_report(report, report_file)
            
            print(f"{Colors.GREEN}✓ Report exported to {report_file}{Colors.END}\n")
            
        except Exception as e:
            print(f"{Colors.RED}✗ Report generation failed: {e}{Colors.END}\n")
    
    def display_review_item(self, item: ReviewItem):
        """
        Display review item with full context
        
        Args:
            item: ReviewItem to display
        """
        print(f"\n{Colors.BOLD}{'─'*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📋 Review Item: {item.review_id}{Colors.END}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.END}\n")
        
        # Document info
        print(f"{Colors.BOLD}Document:{Colors.END} {item.document_id}")
        print(f"{Colors.BOLD}Location:{Colors.END} {item.slide} - {item.location}\n")
        
        # Violation details
        print(f"{Colors.BOLD}Check Type:{Colors.END} {Colors.CYAN}{item.check_type}{Colors.END}")
        print(f"{Colors.BOLD}Severity:{Colors.END} {self._format_severity(item.severity)}")
        print(f"{Colors.BOLD}Predicted Violation:{Colors.END} {self._format_bool(item.predicted_violation)}")
        
        # Confidence with color coding
        confidence_color = self._get_confidence_color(item.confidence)
        print(f"{Colors.BOLD}Confidence:{Colors.END} {confidence_color}{item.confidence}%{Colors.END}")
        print(f"{Colors.BOLD}Priority Score:{Colors.END} {item.priority_score:.2f}\n")
        
        # Rule
        print(f"{Colors.BOLD}Rule:{Colors.END}")
        print(f"  {item.rule}\n")
        
        # AI Reasoning
        print(f"{Colors.BOLD}AI Reasoning:{Colors.END}")
        print(f"  {item.ai_reasoning}\n")
        
        # Evidence
        print(f"{Colors.BOLD}Evidence:{Colors.END}")
        print(f"  {item.evidence}\n")
        
        print(f"{Colors.BOLD}{'─'*70}{Colors.END}\n")
    
    def show_status(self):
        """Display queue status and statistics"""
        stats = self.manager.get_queue_stats()
        
        print(f"\n{Colors.BOLD}{'─'*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📊 Queue Status{Colors.END}")
        print(f"{Colors.BOLD}{'─'*70}{Colors.END}\n")
        
        # Queue counts
        total = stats.total_pending + stats.total_in_review + stats.total_reviewed
        print(f"{Colors.BOLD}Queue Summary:{Colors.END}")
        print(f"  Pending:    {Colors.YELLOW}{stats.total_pending:>4}{Colors.END}")
        print(f"  In Review:  {Colors.CYAN}{stats.total_in_review:>4}{Colors.END}")
        print(f"  Reviewed:   {Colors.GREEN}{stats.total_reviewed:>4}{Colors.END}")
        print(f"  Total:      {Colors.BOLD}{total:>4}{Colors.END}\n")
        
        # Completion percentage
        if total > 0:
            completion = (stats.total_reviewed / total) * 100
            print(f"{Colors.BOLD}Completion:{Colors.END} {completion:.1f}%")
            self._show_progress_bar(stats.total_reviewed, total)
            print()
        
        # Average confidence
        if stats.total_pending > 0:
            print(f"{Colors.BOLD}Avg Confidence:{Colors.END} {stats.avg_confidence:.1f}%\n")
        
        # By check type
        if stats.by_check_type:
            print(f"{Colors.BOLD}By Check Type:{Colors.END}")
            for check_type, count in sorted(stats.by_check_type.items()):
                print(f"  {check_type:<15} {count:>4}")
            print()
        
        # By severity
        if stats.by_severity:
            print(f"{Colors.BOLD}By Severity:{Colors.END}")
            for severity, count in sorted(stats.by_severity.items()):
                severity_formatted = self._format_severity(severity)
                print(f"  {severity_formatted:<25} {count:>4}")
            print()
        
        # Oldest pending
        if stats.oldest_pending_age_hours > 0:
            print(f"{Colors.BOLD}Oldest Pending:{Colors.END} {stats.oldest_pending_age_hours:.1f} hours\n")
        
        print(f"{Colors.BOLD}{'─'*70}{Colors.END}\n")
    
    def show_progress(self, completed: int, total: int):
        """
        Show progress indicator
        
        Args:
            completed: Number of completed reviews
            total: Total number of reviews
        """
        if total == 0:
            return
        
        percentage = (completed / total) * 100
        print(f"{Colors.BOLD}Progress:{Colors.END} {completed}/{total} ({percentage:.1f}%)")
        self._show_progress_bar(completed, total)
        print()
    
    def _show_progress_bar(self, completed: int, total: int, width: int = 50):
        """
        Display a progress bar
        
        Args:
            completed: Number of completed items
            total: Total number of items
            width: Width of progress bar in characters
        """
        if total == 0:
            return
        
        filled = int(width * completed / total)
        bar = '█' * filled + '░' * (width - filled)
        percentage = (completed / total) * 100
        
        print(f"  [{Colors.GREEN}{bar}{Colors.END}] {percentage:.1f}%")
    
    def _show_session_summary(self):
        """Display summary of review session"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📊 Session Summary{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
        
        print(f"Reviewer: {Colors.BOLD}{self.reviewer_id}{Colors.END}\n")
        
        print(f"{Colors.BOLD}Reviews Completed:{Colors.END} {self.session_stats['reviewed']}")
        print(f"  Approved: {Colors.GREEN}{self.session_stats['approved']}{Colors.END}")
        print(f"  Rejected: {Colors.YELLOW}{self.session_stats['rejected']}{Colors.END}")
        print(f"  Skipped:  {Colors.CYAN}{self.session_stats['skipped']}{Colors.END}\n")
        
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    def _show_help(self):
        """Display help information"""
        print(f"\n{Colors.BOLD}Available Commands:{Colors.END}\n")
        print(f"  {Colors.GREEN}next{Colors.END}")
        print(f"    Display next pending review with full context\n")
        print(f"  {Colors.GREEN}approve [--notes 'text']{Colors.END}")
        print(f"    Approve current review (notes optional)\n")
        print(f"  {Colors.GREEN}reject --notes 'text'{Colors.END}")
        print(f"    Reject current review (notes required)\n")
        print(f"  {Colors.GREEN}skip{Colors.END}")
        print(f"    Skip current review and move to next\n")
        print(f"  {Colors.GREEN}status{Colors.END}")
        print(f"    Show queue status and statistics\n")
        print(f"  {Colors.GREEN}metrics [--days N] [--report]{Colors.END}")
        print(f"    Show review metrics and performance statistics")
        print(f"    Options: --days N (default: 30), --report (generate full report)\n")
        print(f"  {Colors.GREEN}batch{Colors.END}")
        print(f"    Batch review operations (use 'help batch' for details)\n")
        print(f"  {Colors.GREEN}quit{Colors.END}")
        print(f"    Exit review session\n")
    
    def _show_batch_help(self):
        """Display batch command help"""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}Batch Review Operations{Colors.END}")
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
        
        print(f"{Colors.BOLD}Usage:{Colors.END}")
        print(f"  batch --<selection> --action=<approve|reject> [--notes 'text'] [--limit=N]\n")
        
        print(f"{Colors.BOLD}Selection Criteria (choose one):{Colors.END}\n")
        
        print(f"  {Colors.GREEN}--check-type=TYPE{Colors.END}")
        print(f"    Select all items with specific check type")
        print(f"    Example: batch --check-type=STRUCTURE --action=approve\n")
        
        print(f"  {Colors.GREEN}--document=DOC_ID{Colors.END}")
        print(f"    Select all items from specific document")
        print(f"    Example: batch --document=doc_001 --action=reject --notes 'Test document'\n")
        
        print(f"  {Colors.GREEN}--severity=LEVEL{Colors.END}")
        print(f"    Select all items with specific severity (CRITICAL, HIGH, MEDIUM, LOW)")
        print(f"    Example: batch --severity=LOW --action=approve\n")
        
        print(f"  {Colors.GREEN}--min-confidence=N --max-confidence=M{Colors.END}")
        print(f"    Select items within confidence range")
        print(f"    Example: batch --min-confidence=60 --max-confidence=70 --action=reject --notes 'Low confidence'\n")
        
        print(f"  {Colors.GREEN}--similar{Colors.END}")
        print(f"    Select items similar to current review (requires active review)")
        print(f"    Example: batch --similar --threshold=0.85 --action=approve\n")
        
        print(f"{Colors.BOLD}Required Parameters:{Colors.END}\n")
        
        print(f"  {Colors.GREEN}--action=<approve|reject>{Colors.END}")
        print(f"    Action to perform on all selected items\n")
        
        print(f"{Colors.BOLD}Optional Parameters:{Colors.END}\n")
        
        print(f"  {Colors.GREEN}--notes 'text'{Colors.END}")
        print(f"    Reviewer notes (required for reject action)\n")
        
        print(f"  {Colors.GREEN}--limit=N{Colors.END}")
        print(f"    Limit number of items to process\n")
        
        print(f"  {Colors.GREEN}--threshold=0.85{Colors.END}")
        print(f"    Similarity threshold for --similar (0-1, default: 0.85)\n")
        
        print(f"{Colors.BOLD}Examples:{Colors.END}\n")
        
        print(f"  # Approve all STRUCTURE checks")
        print(f"  batch --check-type=STRUCTURE --action=approve\n")
        
        print(f"  # Reject all items from a test document")
        print(f"  batch --document=test_doc --action=reject --notes 'Test data'\n")
        
        print(f"  # Approve all LOW severity items")
        print(f"  batch --severity=LOW --action=approve --limit=10\n")
        
        print(f"  # Reject low confidence items")
        print(f"  batch --min-confidence=50 --max-confidence=65 --action=reject --notes 'Too uncertain'\n")
        
        print(f"  # Approve similar items to current review")
        print(f"  batch --similar --action=approve --notes 'Same pattern'\n")
        
        print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    def _extract_notes(self, command: str) -> str:
        """
        Extract notes from command string
        
        Args:
            command: Full command string
            
        Returns:
            Extracted notes or empty string
        """
        # Look for --notes flag
        if '--notes' in command:
            parts = command.split('--notes', 1)
            if len(parts) > 1:
                notes = parts[1].strip()
                # Remove quotes if present
                if notes.startswith('"') and notes.endswith('"'):
                    notes = notes[1:-1]
                elif notes.startswith("'") and notes.endswith("'"):
                    notes = notes[1:-1]
                return notes
        
        return ""
    
    def _format_severity(self, severity: str) -> str:
        """Format severity with color"""
        severity_upper = severity.upper()
        if severity_upper == 'CRITICAL':
            return f"{Colors.RED}{Colors.BOLD}{severity}{Colors.END}"
        elif severity_upper == 'HIGH':
            return f"{Colors.RED}{severity}{Colors.END}"
        elif severity_upper == 'MEDIUM':
            return f"{Colors.YELLOW}{severity}{Colors.END}"
        else:
            return f"{Colors.CYAN}{severity}{Colors.END}"
    
    def _format_bool(self, value: bool) -> str:
        """Format boolean with color"""
        if value:
            return f"{Colors.RED}Yes{Colors.END}"
        else:
            return f"{Colors.GREEN}No{Colors.END}"
    
    def _get_confidence_color(self, confidence: int) -> str:
        """Get color for confidence score"""
        if confidence >= 80:
            return Colors.GREEN
        elif confidence >= 60:
            return Colors.YELLOW
        else:
            return Colors.RED


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='Human-in-the-Loop Review CLI for Compliance Checker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive review session
  python review.py

  # View next pending review
  python review.py next

  # Approve current review
  python review.py approve --notes "Confirmed violation"

  # Reject current review
  python review.py reject --notes "False positive - example only"

  # Skip current review
  python review.py skip

  # Show queue status
  python review.py status

  # Show review metrics
  python review.py metrics

  # Show metrics for last 7 days
  python review.py metrics --days 7

  # Generate full performance report
  python review.py metrics --report

  # Batch approve all STRUCTURE checks
  python review.py batch --check-type=STRUCTURE --action=approve

  # Batch reject items from a document
  python review.py batch --document=doc_001 --action=reject --notes "Test data"

  # Batch approve low severity items
  python review.py batch --severity=LOW --action=approve --limit=10
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['next', 'approve', 'reject', 'skip', 'status', 'metrics', 'batch', 'export', 'report', 'interactive'],
        default='interactive',
        help='Command to execute (default: interactive)'
    )
    
    parser.add_argument(
        '--reviewer-id',
        default='default_reviewer',
        help='Reviewer ID (default: default_reviewer)'
    )
    
    parser.add_argument(
        '--queue-file',
        default='review_queue.json',
        help='Path to review queue file (default: review_queue.json)'
    )
    
    parser.add_argument(
        '--notes',
        default='',
        help='Reviewer notes for approve/reject commands'
    )
    
    # Batch command arguments
    parser.add_argument(
        '--check-type',
        help='Batch: Select items by check type'
    )
    
    parser.add_argument(
        '--document',
        help='Batch: Select items by document ID'
    )
    
    parser.add_argument(
        '--severity',
        help='Batch: Select items by severity level'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=int,
        help='Batch: Minimum confidence score'
    )
    
    parser.add_argument(
        '--max-confidence',
        type=int,
        help='Batch: Maximum confidence score'
    )
    
    parser.add_argument(
        '--similar',
        action='store_true',
        help='Batch: Select items similar to current review'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Batch: Similarity threshold (0-1, default: 0.85)'
    )
    
    parser.add_argument(
        '--action',
        choices=['approve', 'reject'],
        help='Batch: Action to perform (approve or reject)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Batch: Limit number of items to process'
    )
    
    # Metrics command arguments
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Metrics: Number of days to include in analysis (default: 30)'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Metrics: Generate and export full performance report'
    )
    
    # Export command arguments
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Export: Output format (json or csv, default: json)'
    )
    
    parser.add_argument(
        '--output',
        help='Export: Output file path'
    )
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = ReviewCLI(reviewer_id=args.reviewer_id, queue_file=args.queue_file)
    
    # Execute command
    if args.command == 'interactive' or args.command is None:
        cli.start_interactive_session()
    elif args.command == 'next':
        cli.cmd_next()
    elif args.command == 'approve':
        cli.cmd_approve(args.notes)
    elif args.command == 'reject':
        cli.cmd_reject(args.notes)
    elif args.command == 'skip':
        cli.cmd_skip()
    elif args.command == 'status':
        cli.show_status()
    elif args.command == 'metrics':
        # Build metrics command string from arguments
        metrics_cmd = 'metrics'
        if args.days:
            metrics_cmd += f' --days {args.days}'
        if args.report:
            metrics_cmd += ' --report'
        cli.cmd_metrics(metrics_cmd)
    elif args.command == 'batch':
        # Build batch command string from arguments
        batch_cmd = 'batch'
        if args.check_type:
            batch_cmd += f' --check-type={args.check_type}'
        if args.document:
            batch_cmd += f' --document={args.document}'
        if args.severity:
            batch_cmd += f' --severity={args.severity}'
        if args.min_confidence:
            batch_cmd += f' --min-confidence={args.min_confidence}'
        if args.max_confidence:
            batch_cmd += f' --max-confidence={args.max_confidence}'
        if args.similar:
            batch_cmd += f' --similar --threshold={args.threshold}'
        if args.action:
            batch_cmd += f' --action={args.action}'
        if args.notes:
            batch_cmd += f' --notes "{args.notes}"'
        if args.limit:
            batch_cmd += f' --limit={args.limit}'
        
        cli.cmd_batch(batch_cmd)
    elif args.command == 'export':
        # Build export command string from arguments
        export_cmd = 'export'
        if hasattr(args, 'format') and args.format:
            export_cmd += f' --format={args.format}'
        if hasattr(args, 'output') and args.output:
            export_cmd += f' --output={args.output}'
        cli.cmd_export(export_cmd)
    elif args.command == 'report':
        # Build report command string from arguments
        report_cmd = 'report'
        if args.days:
            report_cmd += f' --days={args.days}'
        cli.cmd_report(report_cmd)


if __name__ == "__main__":
    main()
