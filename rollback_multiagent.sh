#!/bin/bash
# Multi-Agent System Rollback Script
# This script restores the previous system state from a backup
# Usage: ./rollback_multiagent.sh [backup_dir] [--force]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR=""
FORCE_MODE=false
PYTHON_CMD="python3"
ROLLBACK_LOG="rollback_$(date +%Y%m%d_%H%M%S).log"
PRE_ROLLBACK_BACKUP=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_MODE=true
            shift
            ;;
        *)
            if [ -z "$BACKUP_DIR" ]; then
                BACKUP_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$ROLLBACK_LOG"
    
    case $level in
        ERROR)
            print_error "$message"
            ;;
        WARNING)
            print_warning "$message"
            ;;
        SUCCESS)
            print_success "$message"
            ;;
        *)
            print_info "$message"
            ;;
    esac
}

# Find the latest backup
find_latest_backup() {
    print_header "Finding Latest Backup"
    
    if [ ! -d "backups" ]; then
        log_message ERROR "No backups directory found"
        return 1
    fi
    
    # Find all backup directories
    local backup_dirs=$(find backups -maxdepth 1 -type d -name "deployment_*" -o -name "backup_*" | sort -r)
    
    if [ -z "$backup_dirs" ]; then
        log_message ERROR "No backup directories found in backups/"
        return 1
    fi
    
    local count=$(echo "$backup_dirs" | wc -l)
    log_message INFO "Found $count backup(s)"
    
    # Display available backups
    echo ""
    print_info "Available backups:"
    local i=1
    echo "$backup_dirs" | head -5 | while read -r dir; do
        local timestamp=$(stat -c %y "$dir" 2>/dev/null || stat -f "%Sm" "$dir" 2>/dev/null || echo "Unknown")
        echo "  $i. $(basename $dir) (Created: $timestamp)"
        i=$((i+1))
    done
    echo ""
    
    # Return the latest backup
    BACKUP_DIR=$(echo "$backup_dirs" | head -1)
    log_message SUCCESS "Latest backup: $BACKUP_DIR"
    return 0
}

# Confirm rollback with user
confirm_rollback() {
    print_header "Rollback Confirmation"
    
    echo ""
    print_warning "This will restore the system from backup:"
    echo "  Backup: $BACKUP_DIR"
    echo ""
    print_warning "Current files will be overwritten!"
    echo ""
    
    if [ "$FORCE_MODE" = true ]; then
        log_message INFO "Rollback confirmed (forced)"
        return 0
    fi
    
    read -p "Do you want to proceed? (yes/no): " response
    
    if [ "$response" = "yes" ]; then
        log_message INFO "Rollback confirmed by user"
        return 0
    else
        log_message WARNING "Rollback cancelled by user"
        return 1
    fi
}

# Test backup integrity
test_backup_integrity() {
    print_header "Validating Backup Integrity"
    
    # Check if backup directory exists
    if [ ! -d "$BACKUP_DIR" ]; then
        log_message ERROR "Backup directory not found: $BACKUP_DIR"
        return 1
    fi
    
    log_message SUCCESS "Backup directory exists"
    
    # Check for manifest file
    if [ -f "$BACKUP_DIR/manifest.json" ]; then
        log_message SUCCESS "Backup manifest found"
        
        # Parse manifest if jq is available
        if command -v jq &> /dev/null; then
            local timestamp=$(jq -r '.timestamp' "$BACKUP_DIR/manifest.json" 2>/dev/null || echo "Unknown")
            local python_ver=$(jq -r '.python_version' "$BACKUP_DIR/manifest.json" 2>/dev/null || echo "Unknown")
            log_message INFO "  Backup timestamp: $timestamp"
            log_message INFO "  Python version: $python_ver"
        fi
    else
        log_message WARNING "No manifest file found (older backup format)"
    fi
    
    # Check for critical files
    local critical_files=("hybrid_config.json")
    local all_exist=true
    
    for file in "${critical_files[@]}"; do
        if [ -f "$BACKUP_DIR/$file" ]; then
            log_message SUCCESS "Found critical file: $file"
        else
            log_message WARNING "Missing critical file: $file"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = false ]; then
        log_message WARNING "Backup may be incomplete"
        if [ "$FORCE_MODE" = false ]; then
            echo ""
            read -p "Continue anyway? (yes/no): " response
            if [ "$response" != "yes" ]; then
                return 1
            fi
        fi
    fi
    
    log_message SUCCESS "Backup integrity check passed"
    return 0
}

# Stop running processes
stop_running_processes() {
    print_header "Stopping Running Processes"
    
    # Check for running Python processes
    local python_pids=$(pgrep -f "python.*check" || true)
    
    if [ -n "$python_pids" ]; then
        log_message WARNING "Found Python processes running: $python_pids"
        echo ""
        print_warning "The following Python processes are running:"
        ps -p $python_pids -o pid,cmd || true
        echo ""
        
        if [ "$FORCE_MODE" = false ]; then
            read -p "Stop these processes? (yes/no): " response
            if [ "$response" = "yes" ]; then
                for pid in $python_pids; do
                    if kill -9 $pid 2>/dev/null; then
                        log_message SUCCESS "Stopped process: $pid"
                    else
                        log_message WARNING "Failed to stop process: $pid"
                    fi
                done
            else
                log_message WARNING "Processes not stopped - rollback may fail if files are in use"
            fi
        fi
    else
        log_message SUCCESS "No Python processes found running"
    fi
}

# Create pre-rollback backup
create_pre_rollback_backup() {
    print_header "Creating Pre-Rollback Backup"
    
    PRE_ROLLBACK_BACKUP="backups/pre_rollback_$(date +%Y%m%d_%H%M%S)"
    
    if ! mkdir -p "$PRE_ROLLBACK_BACKUP"; then
        log_message ERROR "Failed to create pre-rollback backup directory"
        return 1
    fi
    
    log_message SUCCESS "Created pre-rollback backup directory: $PRE_ROLLBACK_BACKUP"
    
    # Files to backup
    local files_to_backup=("hybrid_config.json" ".env" "review_queue.json")
    local dirs_to_backup=("checkpoints" "audit_logs")
    
    # Backup files
    for file in "${files_to_backup[@]}"; do
        if [ -f "$file" ]; then
            if cp "$file" "$PRE_ROLLBACK_BACKUP/"; then
                log_message SUCCESS "Backed up current: $file"
            else
                log_message WARNING "Failed to backup: $file"
            fi
        fi
    done
    
    # Backup directories
    for dir in "${dirs_to_backup[@]}"; do
        if [ -d "$dir" ]; then
            if cp -r "$dir" "$PRE_ROLLBACK_BACKUP/"; then
                log_message SUCCESS "Backed up current: $dir/"
            else
                log_message WARNING "Failed to backup: $dir/"
            fi
        fi
    done
    
    log_message SUCCESS "Pre-rollback backup completed: $PRE_ROLLBACK_BACKUP"
    return 0
}

# Restore configuration
restore_configuration() {
    print_header "Restoring Configuration"
    
    # Restore hybrid_config.json
    if [ -f "$BACKUP_DIR/hybrid_config.json" ]; then
        if cp "$BACKUP_DIR/hybrid_config.json" "hybrid_config.json"; then
            log_message SUCCESS "Restored: hybrid_config.json"
            
            # Disable multi-agent mode using Python
            $PYTHON_CMD -c "
import json
try:
    with open('hybrid_config.json', 'r') as f:
        config = json.load(f)
    
    if 'multi_agent' in config:
        config['multi_agent']['enabled'] = False
        with open('hybrid_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print('Disabled multi-agent mode')
except Exception as e:
    print(f'Error: {e}')
" >> "$ROLLBACK_LOG" 2>&1
            
            log_message SUCCESS "Disabled multi-agent mode in configuration"
        else
            log_message ERROR "Failed to restore configuration"
            return 1
        fi
    else
        log_message WARNING "No configuration file in backup"
    fi
    
    # Restore .env
    if [ -f "$BACKUP_DIR/.env" ]; then
        if cp "$BACKUP_DIR/.env" ".env"; then
            log_message SUCCESS "Restored: .env"
        else
            log_message WARNING "Failed to restore .env"
        fi
    fi
    
    # Restore review queue
    if [ -f "$BACKUP_DIR/review_queue.json" ]; then
        if cp "$BACKUP_DIR/review_queue.json" "review_queue.json"; then
            log_message SUCCESS "Restored: review_queue.json"
        else
            log_message WARNING "Failed to restore review queue"
        fi
    fi
    
    # Restore audit logs
    if [ -d "$BACKUP_DIR/audit_logs" ]; then
        rm -rf "audit_logs" 2>/dev/null || true
        if cp -r "$BACKUP_DIR/audit_logs" "audit_logs"; then
            log_message SUCCESS "Restored: audit_logs/"
        else
            log_message WARNING "Failed to restore audit logs"
        fi
    fi
    
    # Restore checkpoints
    if [ -d "$BACKUP_DIR/checkpoints" ]; then
        rm -rf "checkpoints" 2>/dev/null || true
        if cp -r "$BACKUP_DIR/checkpoints" "checkpoints"; then
            log_message SUCCESS "Restored: checkpoints/"
        else
            log_message WARNING "Failed to restore checkpoints"
        fi
    fi
    
    log_message SUCCESS "Configuration restoration complete"
    return 0
}

# Restore files
restore_files() {
    print_header "Restoring Files from Backup"
    
    # Count files to restore
    local file_count=$(find "$BACKUP_DIR" -type f ! -name "manifest.json" | wc -l)
    log_message INFO "Found $file_count file(s) to restore"
    
    local restored=0
    local failed=0
    
    # Restore all files except manifest
    find "$BACKUP_DIR" -type f ! -name "manifest.json" | while read -r file; do
        # Get relative path
        local rel_path="${file#$BACKUP_DIR/}"
        
        # Create directory if needed
        local target_dir=$(dirname "$rel_path")
        if [ "$target_dir" != "." ] && [ ! -d "$target_dir" ]; then
            mkdir -p "$target_dir"
        fi
        
        # Copy file
        if cp "$file" "$rel_path"; then
            log_message SUCCESS "Restored: $rel_path"
            restored=$((restored+1))
        else
            log_message ERROR "Failed to restore: $rel_path"
            failed=$((failed+1))
        fi
    done
    
    echo ""
    log_message INFO "Restoration complete: $restored succeeded, $failed failed"
    
    if [ $failed -gt 0 ]; then
        log_message WARNING "Some files failed to restore"
        return 1
    fi
    
    return 0
}

# Test rollback success
test_rollback_success() {
    print_header "Verifying Rollback Success"
    
    # Check critical files exist
    local critical_files=("check.py" "hybrid_config.json")
    local all_exist=true
    
    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            log_message SUCCESS "Verified: $file exists"
        else
            log_message ERROR "Missing: $file"
            all_exist=false
        fi
    done
    
    if [ "$all_exist" = false ]; then
        log_message ERROR "Rollback verification failed - missing critical files"
        return 1
    fi
    
    # Verify configuration
    if [ -f "hybrid_config.json" ]; then
        if $PYTHON_CMD -c "import json; json.load(open('hybrid_config.json'))" 2>/dev/null; then
            log_message SUCCESS "Configuration file is valid JSON"
            
            # Check multi-agent mode
            local ma_enabled=$($PYTHON_CMD -c "import json; config=json.load(open('hybrid_config.json')); print(config.get('multi_agent', {}).get('enabled', 'not_present'))" 2>/dev/null)
            
            if [ "$ma_enabled" = "False" ]; then
                log_message SUCCESS "Multi-agent mode is disabled"
            elif [ "$ma_enabled" = "not_present" ]; then
                log_message SUCCESS "Multi-agent configuration not present (pre-migration state)"
            else
                log_message WARNING "Multi-agent mode is still enabled"
            fi
        else
            log_message ERROR "Configuration file is invalid"
            return 1
        fi
    fi
    
    # Test basic functionality
    if [ -f "check.py" ]; then
        log_message INFO "Testing basic system functionality..."
        if $PYTHON_CMD check.py --help > /dev/null 2>&1; then
            log_message SUCCESS "Basic system test passed"
        else
            log_message WARNING "Basic system test failed"
        fi
    fi
    
    log_message SUCCESS "Rollback verification complete"
    return 0
}

# Display rollback summary
show_rollback_summary() {
    local success=$1
    
    print_header "Rollback Summary"
    
    echo ""
    print_info "Rollback Status:"
    if [ "$success" = true ]; then
        print_success "  SUCCESS"
    else
        print_error "  FAILED"
    fi
    
    echo ""
    print_info "Rollback Log: $ROLLBACK_LOG"
    print_info "Restored From: $BACKUP_DIR"
    
    if [ -n "$PRE_ROLLBACK_BACKUP" ]; then
        print_info "Pre-Rollback Backup: $PRE_ROLLBACK_BACKUP"
        print_warning "  (Use this to restore if rollback was incorrect)"
    fi
    
    if [ "$success" = true ]; then
        echo ""
        print_success "System has been restored to previous state"
        echo ""
        print_info "Next Steps:"
        echo "  1. Review the rollback log for any warnings"
        echo "  2. Test the system: python check.py exemple.json"
        echo "  3. Verify your configuration in hybrid_config.json"
        echo "  4. Check that all expected files are present"
        echo ""
        print_info "If you need to restore the multi-agent system:"
        echo "  ./deploy_multiagent.sh"
    else
        echo ""
        print_error "Rollback failed!"
        echo ""
        print_info "Recovery Options:"
        echo "  1. Review the rollback log: $ROLLBACK_LOG"
        echo "  2. Try rollback again with --force flag"
        echo "  3. Manually restore files from: $BACKUP_DIR"
        if [ -n "$PRE_ROLLBACK_BACKUP" ]; then
            echo "  4. Restore pre-rollback state from: $PRE_ROLLBACK_BACKUP"
        fi
    fi
    
    echo ""
    echo "========================================"
}

# Main rollback flow
main() {
    clear
    print_header "Multi-Agent System Rollback"
    echo ""
    log_message INFO "Starting rollback..."
    log_message INFO "Rollback log: $ROLLBACK_LOG"
    echo ""
    sleep 1
    
    # Step 1: Find backup to restore
    if [ -z "$BACKUP_DIR" ]; then
        if ! find_latest_backup; then
            show_rollback_summary false
            exit 1
        fi
    else
        # Use specified backup directory
        if [ ! -d "$BACKUP_DIR" ]; then
            log_message ERROR "Specified backup directory not found: $BACKUP_DIR"
            show_rollback_summary false
            exit 1
        fi
        log_message INFO "Using specified backup: $BACKUP_DIR"
    fi
    echo ""
    
    # Step 2: Validate backup integrity
    if ! test_backup_integrity; then
        log_message ERROR "Backup integrity check failed"
        show_rollback_summary false
        exit 1
    fi
    echo ""
    
    # Step 3: Confirm rollback
    if ! confirm_rollback; then
        log_message INFO "Rollback cancelled"
        exit 0
    fi
    echo ""
    
    # Step 4: Stop running processes
    stop_running_processes
    echo ""
    
    # Step 5: Create pre-rollback backup
    if ! create_pre_rollback_backup; then
        log_message WARNING "Failed to create pre-rollback backup"
        if [ "$FORCE_MODE" = false ]; then
            read -p "Continue without pre-rollback backup? (yes/no): " response
            if [ "$response" != "yes" ]; then
                log_message INFO "Rollback cancelled"
                exit 0
            fi
        fi
    fi
    echo ""
    
    # Step 6: Restore configuration
    if ! restore_configuration; then
        log_message ERROR "Configuration restoration failed"
        show_rollback_summary false
        exit 1
    fi
    echo ""
    
    # Step 7: Restore files
    if ! restore_files; then
        log_message WARNING "File restoration had errors"
        if [ "$FORCE_MODE" = false ]; then
            show_rollback_summary false
            exit 1
        fi
    fi
    echo ""
    
    # Step 8: Verify rollback success
    if ! test_rollback_success; then
        log_message ERROR "Rollback verification failed"
        show_rollback_summary false
        exit 1
    fi
    echo ""
    
    # Success!
    log_message SUCCESS "Rollback completed successfully"
    show_rollback_summary true
    exit 0
}

# Run main function
main
