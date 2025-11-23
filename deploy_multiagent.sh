#!/bin/bash
# Multi-Agent Compliance Checker Deployment Script
# This script deploys the LangGraph-based multi-agent compliance system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="backups/deployment_$(date +%Y%m%d_%H%M%S)"
PYTHON_CMD="python3"
VENV_DIR="venv"
CONFIG_FILE="hybrid_config.json"
REQUIREMENTS_FILE="requirements.txt"

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

# Check if Python is available
check_python() {
    print_header "Checking Python Installation"
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
    
    # Check Python version (minimum 3.8)
    PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    print_success "Python version is compatible"
}

# Create backup of current system
create_backup() {
    print_header "Creating Backup"
    
    if [ ! -d "backups" ]; then
        mkdir -p backups
        print_info "Created backups directory"
    fi
    
    mkdir -p "$BACKUP_DIR"
    print_info "Backup directory: $BACKUP_DIR"
    
    # Backup configuration files
    if [ -f "$CONFIG_FILE" ]; then
        cp "$CONFIG_FILE" "$BACKUP_DIR/"
        print_success "Backed up $CONFIG_FILE"
    fi
    
    if [ -f ".env" ]; then
        cp ".env" "$BACKUP_DIR/"
        print_success "Backed up .env"
    fi
    
    # Backup review queue
    if [ -f "review_queue.json" ]; then
        cp "review_queue.json" "$BACKUP_DIR/"
        print_success "Backed up review_queue.json"
    fi
    
    # Backup audit logs
    if [ -d "audit_logs" ]; then
        cp -r "audit_logs" "$BACKUP_DIR/"
        print_success "Backed up audit_logs/"
    fi
    
    # Backup checkpoints
    if [ -d "checkpoints" ]; then
        cp -r "checkpoints" "$BACKUP_DIR/"
        print_success "Backed up checkpoints/"
    fi
    
    print_success "Backup completed: $BACKUP_DIR"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_info "Installing Python packages from $REQUIREMENTS_FILE..."
    $PYTHON_CMD -m pip install --upgrade pip
    $PYTHON_CMD -m pip install -r "$REQUIREMENTS_FILE"
    
    print_success "Dependencies installed successfully"
}

# Validate configuration
validate_configuration() {
    print_header "Validating Configuration"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file $CONFIG_FILE not found"
        exit 1
    fi
    
    print_success "Configuration file found: $CONFIG_FILE"
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found"
        if [ -f ".env.example" ]; then
            print_info "Copying .env.example to .env"
            cp .env.example .env
            print_warning "Please configure your API keys in .env file"
        else
            print_warning "Please create .env file with your API keys"
        fi
    else
        print_success ".env file found"
    fi
    
    # Validate configuration structure using Python
    print_info "Validating configuration structure..."
    $PYTHON_CMD -c "
import json
import sys

try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    
    # Check required sections
    required_sections = ['multi_agent', 'agents', 'routing']
    missing = [s for s in required_sections if s not in config]
    
    if missing:
        print(f'Missing required sections: {missing}')
        sys.exit(1)
    
    # Check multi_agent settings
    if not config.get('multi_agent', {}).get('enabled', False):
        print('Warning: multi_agent.enabled is false')
    
    print('Configuration structure is valid')
    sys.exit(0)
    
except json.JSONDecodeError as e:
    print(f'Invalid JSON in configuration file: {e}')
    sys.exit(1)
except Exception as e:
    print(f'Error validating configuration: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Configuration structure is valid"
    else
        print_error "Configuration validation failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    print_info "Running unit tests..."
    
    # Check if pytest is available
    if $PYTHON_CMD -c "import pytest" 2>/dev/null; then
        # Run tests with pytest
        if [ -d "tests" ]; then
            print_info "Running pytest..."
            $PYTHON_CMD -m pytest tests/ -v --tb=short || {
                print_warning "Some tests failed. Continue anyway? (y/n)"
                read -r response
                if [ "$response" != "y" ]; then
                    print_error "Deployment cancelled"
                    exit 1
                fi
            }
        else
            print_warning "tests/ directory not found, skipping pytest"
        fi
    else
        print_warning "pytest not installed, skipping unit tests"
    fi
    
    # Run basic validation test
    print_info "Running basic validation test..."
    if [ -f "exemple.json" ]; then
        $PYTHON_CMD check_multiagent.py exemple.json --show-metrics > /dev/null 2>&1 || {
            print_warning "Validation test failed. Continue anyway? (y/n)"
            read -r response
            if [ "$response" != "y" ]; then
                print_error "Deployment cancelled"
                exit 1
            fi
        }
        print_success "Basic validation test passed"
    else
        print_warning "exemple.json not found, skipping validation test"
    fi
}

# Create required directories
create_directories() {
    print_header "Creating Required Directories"
    
    directories=(
        "checkpoints"
        "audit_logs"
        "monitoring/logs"
        "monitoring/metrics"
        "monitoring/visualizations"
        "backups"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
}

# Enable multi-agent mode in configuration
enable_multiagent() {
    print_header "Enabling Multi-Agent Mode"
    
    print_info "Updating configuration to enable multi-agent mode..."
    
    $PYTHON_CMD -c "
import json

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

# Enable multi-agent mode
if 'multi_agent' not in config:
    config['multi_agent'] = {}

config['multi_agent']['enabled'] = True

# Save updated configuration
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config, f, indent=2)

print('Multi-agent mode enabled in configuration')
"
    
    print_success "Multi-agent mode enabled"
}

# Display deployment summary
display_summary() {
    print_header "Deployment Summary"
    
    echo ""
    print_success "Multi-Agent Compliance Checker deployed successfully!"
    echo ""
    print_info "Backup location: $BACKUP_DIR"
    print_info "Configuration: $CONFIG_FILE"
    print_info "Entry point: check_multiagent.py"
    echo ""
    print_info "Usage:"
    echo "  python check_multiagent.py <json_file> [options]"
    echo ""
    print_info "Examples:"
    echo "  python check_multiagent.py exemple.json"
    echo "  python check_multiagent.py exemple.json --show-metrics"
    echo "  python check_multiagent.py exemple.json --review-mode"
    echo ""
    print_info "Features enabled:"
    echo "  ✓ Multi-agent architecture"
    echo "  ✓ Parallel execution"
    echo "  ✓ State persistence"
    echo "  ✓ Human-in-the-loop integration"
    echo "  ✓ AI-enhanced checking"
    echo "  ✓ Context-aware validation"
    echo ""
    print_info "Next steps:"
    echo "  1. Configure API keys in .env file (if not already done)"
    echo "  2. Review configuration in $CONFIG_FILE"
    echo "  3. Run a test: python check_multiagent.py exemple.json"
    echo "  4. Check monitoring dashboard: python monitoring/dashboard.py"
    echo ""
    print_success "Deployment complete!"
}

# Main deployment flow
main() {
    clear
    print_header "Multi-Agent Compliance Checker Deployment"
    echo ""
    print_info "This script will deploy the LangGraph-based multi-agent system"
    print_info "Press Ctrl+C to cancel at any time"
    echo ""
    sleep 2
    
    # Step 1: Check Python
    check_python
    echo ""
    
    # Step 2: Create backup
    create_backup
    echo ""
    
    # Step 3: Install dependencies
    install_dependencies
    echo ""
    
    # Step 4: Validate configuration
    validate_configuration
    echo ""
    
    # Step 5: Create required directories
    create_directories
    echo ""
    
    # Step 6: Run tests
    print_info "Run tests before deployment? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        run_tests
        echo ""
    else
        print_warning "Skipping tests"
        echo ""
    fi
    
    # Step 7: Enable multi-agent mode
    enable_multiagent
    echo ""
    
    # Step 8: Display summary
    display_summary
}

# Run main function
main
