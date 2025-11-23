#!/usr/bin/env pwsh
# Multi-Agent System Deployment Script
# This script handles the complete deployment of the multi-agent compliance system
# including dependency installation, testing, validation, backup, and deployment

param(
    [switch]$SkipTests = $false,
    [switch]$SkipBackup = $false,
    [switch]$Force = $false,
    [string]$BackupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [string]$PythonCmd = "python"
)

# Color output functions
function Write-Success { param($Message) Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Error { param($Message) Write-Host "✗ $Message" -ForegroundColor Red }
function Write-Info { param($Message) Write-Host "ℹ $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠ $Message" -ForegroundColor Yellow }
function Write-Header { param($Message) Write-Host "`n$('='*70)" -ForegroundColor Yellow; Write-Host $Message -ForegroundColor Yellow; Write-Host "$('='*70)" -ForegroundColor Yellow }

# Error handling
$ErrorActionPreference = "Stop"
$script:DeploymentFailed = $false
$script:DeploymentLog = "deployment_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Add-Content -Path $script:DeploymentLog -Value $logMessage
    
    switch ($Level) {
        "ERROR" { Write-Error $Message }
        "WARNING" { Write-Warning $Message }
        "SUCCESS" { Write-Success $Message }
        default { Write-Info $Message }
    }
}

function Test-PythonInstallation {
    Write-Header "CHECKING PYTHON INSTALLATION"
    
    try {
        $pythonVersion = & $PythonCmd --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Python found: $pythonVersion" "SUCCESS"
            
            # Check Python version (need 3.8+)
            if ($pythonVersion -match "Python (\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                
                if ($major -ge 3 -and $minor -ge 8) {
                    Write-Log "Python version is compatible (3.8+)" "SUCCESS"
                    return $true
                } else {
                    Write-Log "Python version must be 3.8 or higher" "ERROR"
                    return $false
                }
            }
        }
    } catch {
        Write-Log "Python not found. Please install Python 3.8 or higher" "ERROR"
        return $false
    }
    
    return $false
}

function Install-Dependencies {
    Write-Header "INSTALLING DEPENDENCIES"
    
    Write-Log "Upgrading pip..."
    try {
        & $PythonCmd -m pip install --upgrade pip | Out-Null
        Write-Log "pip upgraded successfully" "SUCCESS"
    } catch {
        Write-Log "Failed to upgrade pip: $_" "WARNING"
    }
    
    Write-Log "Installing requirements from requirements.txt..."
    try {
        & $PythonCmd -m pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Log "All dependencies installed successfully" "SUCCESS"
            return $true
        } else {
            Write-Log "Failed to install some dependencies" "ERROR"
            return $false
        }
    } catch {
        Write-Log "Error installing dependencies: $_" "ERROR"
        return $false
    }
}

function Test-Configuration {
    Write-Header "VALIDATING CONFIGURATION"
    
    # Check for required configuration files
    $requiredFiles = @(
        "hybrid_config.json",
        ".env.example"
    )
    
    $allFilesExist = $true
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Log "Found: $file" "SUCCESS"
        } else {
            Write-Log "Missing: $file" "ERROR"
            $allFilesExist = $false
        }
    }
    
    # Check for .env file
    if (Test-Path ".env") {
        Write-Log "Found: .env" "SUCCESS"
    } else {
        Write-Log ".env file not found" "WARNING"
        if (Test-Path ".env.example") {
            Write-Log "Please copy .env.example to .env and configure API keys" "WARNING"
        }
    }
    
    # Validate hybrid_config.json
    if (Test-Path "hybrid_config.json") {
        try {
            $config = Get-Content "hybrid_config.json" -Raw | ConvertFrom-Json
            
            # Check multi_agent section
            if ($config.multi_agent) {
                Write-Log "Multi-agent configuration found" "SUCCESS"
                
                # Check critical settings
                if ($config.multi_agent.enabled -ne $null) {
                    Write-Log "  - multi_agent.enabled: $($config.multi_agent.enabled)" "INFO"
                }
                if ($config.multi_agent.parallel_execution -ne $null) {
                    Write-Log "  - parallel_execution: $($config.multi_agent.parallel_execution)" "INFO"
                }
                if ($config.multi_agent.state_persistence -ne $null) {
                    Write-Log "  - state_persistence: $($config.multi_agent.state_persistence)" "INFO"
                }
            } else {
                Write-Log "Multi-agent configuration section missing" "WARNING"
            }
            
            # Check agents section
            if ($config.agents) {
                $agentCount = ($config.agents | Get-Member -MemberType NoteProperty).Count
                Write-Log "Found $agentCount agent configurations" "SUCCESS"
            }
            
        } catch {
            Write-Log "Error parsing hybrid_config.json: $_" "ERROR"
            return $false
        }
    }
    
    return $allFilesExist
}

function Run-Tests {
    Write-Header "RUNNING TESTS"
    
    if ($SkipTests) {
        Write-Log "Tests skipped by user" "WARNING"
        return $true
    }
    
    # Check if pytest is available
    try {
        & $PythonCmd -m pytest --version | Out-Null
        $hasPytest = ($LASTEXITCODE -eq 0)
    } catch {
        $hasPytest = $false
    }
    
    if (-not $hasPytest) {
        Write-Log "pytest not installed, installing..." "INFO"
        try {
            & $PythonCmd -m pip install pytest
            Write-Log "pytest installed successfully" "SUCCESS"
        } catch {
            Write-Log "Failed to install pytest, skipping tests" "WARNING"
            return $true
        }
    }
    
    # Run critical tests
    $testFiles = @(
        "test_data_models_multiagent.py",
        "test_base_agent.py",
        "test_workflow_builder.py"
    )
    
    $allTestsPassed = $true
    foreach ($testFile in $testFiles) {
        if (Test-Path $testFile) {
            Write-Log "Running $testFile..." "INFO"
            try {
                & $PythonCmd -m pytest $testFile -v
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "$testFile passed" "SUCCESS"
                } else {
                    Write-Log "$testFile failed" "ERROR"
                    $allTestsPassed = $false
                }
            } catch {
                Write-Log "Error running ${testFile}: $($_.Exception.Message)" "ERROR"
                $allTestsPassed = $false
            }
        } else {
            Write-Log "$testFile not found, skipping" "WARNING"
        }
    }
    
    # Run integration tests if available
    if (Test-Path "tests/test_workflow.py") {
        Write-Log "Running integration tests..." "INFO"
        try {
            & $PythonCmd -m pytest tests/test_workflow.py -v
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Integration tests passed" "SUCCESS"
            } else {
                Write-Log "Integration tests failed" "WARNING"
                if (-not $Force) {
                    $allTestsPassed = $false
                }
            }
        } catch {
            Write-Log "Error running integration tests: $_" "WARNING"
        }
    }
    
    if (-not $allTestsPassed -and -not $Force) {
        Write-Log "Tests failed. Use -Force to deploy anyway" "ERROR"
        return $false
    }
    
    return $true
}

function Create-Backup {
    Write-Header "CREATING BACKUP"
    
    if ($SkipBackup) {
        Write-Log "Backup skipped by user" "WARNING"
        return $true
    }
    
    # Create backup directory
    try {
        New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
        Write-Log "Created backup directory: $BackupDir" "SUCCESS"
    } catch {
        Write-Log "Failed to create backup directory: $_" "ERROR"
        return $false
    }
    
    # Files to backup
    $filesToBackup = @(
        "check.py",
        "check_ai.py",
        "check_hybrid.py",
        "hybrid_config.json",
        ".env"
    )
    
    # Directories to backup
    $dirsToBackup = @(
        "agents",
        "tools",
        "monitoring"
    )
    
    # Backup files
    foreach ($file in $filesToBackup) {
        if (Test-Path $file) {
            try {
                Copy-Item $file -Destination $BackupDir -Force
                Write-Log "Backed up: $file" "SUCCESS"
            } catch {
                Write-Log "Failed to backup ${file}: $($_.Exception.Message)" "WARNING"
            }
        }
    }
    
    # Backup directories
    foreach ($dir in $dirsToBackup) {
        if (Test-Path $dir) {
            try {
                Copy-Item $dir -Destination $BackupDir -Recurse -Force
                Write-Log "Backed up: $dir/" "SUCCESS"
            } catch {
                Write-Log "Failed to backup ${dir}: $($_.Exception.Message)" "WARNING"
            }
        }
    }
    
    # Create backup manifest
    $manifest = @{
        timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        files = $filesToBackup | Where-Object { Test-Path $_ }
        directories = $dirsToBackup | Where-Object { Test-Path $_ }
        python_version = & $PythonCmd --version 2>&1
    }
    
    $manifest | ConvertTo-Json | Out-File "$BackupDir/manifest.json" -Encoding UTF8
    Write-Log "Backup manifest created" "SUCCESS"
    
    Write-Log "Backup completed successfully in: $BackupDir" "SUCCESS"
    return $true
}

function Deploy-MultiAgentSystem {
    Write-Header "DEPLOYING MULTI-AGENT SYSTEM"
    
    # Check required files exist
    $requiredFiles = @(
        "check_multiagent.py",
        "workflow_builder.py",
        "data_models_multiagent.py",
        "agents/base_agent.py",
        "agents/supervisor_agent.py"
    )
    
    $allFilesExist = $true
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Log "Verified: $file" "SUCCESS"
        } else {
            Write-Log "Missing required file: $file" "ERROR"
            $allFilesExist = $false
        }
    }
    
    if (-not $allFilesExist) {
        Write-Log "Deployment cannot proceed - missing required files" "ERROR"
        return $false
    }
    
    # Create necessary directories
    $directories = @(
        "checkpoints",
        "monitoring/logs",
        "monitoring/metrics",
        "monitoring/visualizations",
        "audit_logs"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            try {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
                Write-Log "Created directory: $dir" "SUCCESS"
            } catch {
                Write-Log "Failed to create directory ${dir}: $($_.Exception.Message)" "WARNING"
            }
        } else {
            Write-Log "Directory exists: $dir" "INFO"
        }
    }
    
    # Enable multi-agent mode in configuration
    if (Test-Path "hybrid_config.json") {
        try {
            $config = Get-Content "hybrid_config.json" -Raw | ConvertFrom-Json
            
            if ($config.multi_agent) {
                $config.multi_agent.enabled = $true
                Write-Log "Enabled multi-agent mode in configuration" "SUCCESS"
            } else {
                Write-Log "multi_agent section not found in configuration" "WARNING"
            }
            
            # Save updated configuration
            $config | ConvertTo-Json -Depth 10 | Out-File "hybrid_config.json" -Encoding UTF8
            Write-Log "Configuration updated" "SUCCESS"
            
        } catch {
            Write-Log "Failed to update configuration: $_" "ERROR"
            return $false
        }
    }
    
    # Test the multi-agent system with exemple.json
    if (Test-Path "exemple.json") {
        Write-Log "Testing multi-agent system with exemple.json..." "INFO"
        try {
            & $PythonCmd check_multiagent.py exemple.json --show-metrics
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Multi-agent system test passed" "SUCCESS"
            } else {
                Write-Log "Multi-agent system test completed with violations (expected)" "INFO"
            }
        } catch {
            Write-Log "Error testing multi-agent system: $_" "WARNING"
            if (-not $Force) {
                return $false
            }
        }
    } else {
        Write-Log "exemple.json not found, skipping system test" "WARNING"
    }
    
    Write-Log "Multi-agent system deployed successfully" "SUCCESS"
    return $true
}

function Show-DeploymentSummary {
    Write-Header "DEPLOYMENT SUMMARY"
    
    Write-Host ""
    Write-Host "Deployment Status:" -ForegroundColor Cyan
    if ($script:DeploymentFailed) {
        Write-Host "  ✗ FAILED" -ForegroundColor Red
    } else {
        Write-Host "  ✓ SUCCESS" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Deployment Log: $script:DeploymentLog" -ForegroundColor Cyan
    
    if (-not $SkipBackup) {
        Write-Host "Backup Location: $BackupDir" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "  1. Review the deployment log for any warnings" -ForegroundColor White
    Write-Host "  2. Configure .env file with your API keys (if not done)" -ForegroundColor White
    Write-Host "  3. Test the system: python check_multiagent.py exemple.json" -ForegroundColor White
    Write-Host "  4. Review documentation in docs/MULTI_AGENT_ARCHITECTURE.md" -ForegroundColor White
    
    if (-not $script:DeploymentFailed) {
        Write-Host ""
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "  python check_multiagent.py [json_file] [options]" -ForegroundColor White
        Write-Host ""
        Write-Host "Options:" -ForegroundColor Yellow
        Write-Host "  --hybrid-mode=on|off    Enable/disable AI+Rules hybrid mode" -ForegroundColor White
        Write-Host "  --show-metrics          Display performance metrics" -ForegroundColor White
        Write-Host "  --review-mode           Enter interactive review mode" -ForegroundColor White
        Write-Host ""
        Write-Host "To rollback:" -ForegroundColor Yellow
        Write-Host "  .\rollback_multiagent.ps1 -BackupDir $BackupDir" -ForegroundColor White
    }
    
    Write-Host ""
    Write-Host "$('='*70)" -ForegroundColor Yellow
}

# Main deployment flow
function Main {
    Write-Header "MULTI-AGENT SYSTEM DEPLOYMENT"
    Write-Log "Starting deployment..." "INFO"
    Write-Log "Deployment log: $script:DeploymentLog" "INFO"
    
    # Step 1: Check Python installation
    if (-not (Test-PythonInstallation)) {
        $script:DeploymentFailed = $true
        Show-DeploymentSummary
        exit 1
    }
    
    # Step 2: Install dependencies
    if (-not (Install-Dependencies)) {
        $script:DeploymentFailed = $true
        Show-DeploymentSummary
        exit 1
    }
    
    # Step 3: Validate configuration
    if (-not (Test-Configuration)) {
        Write-Log "Configuration validation failed" "WARNING"
        if (-not $Force) {
            $script:DeploymentFailed = $true
            Show-DeploymentSummary
            exit 1
        }
    }
    
    # Step 4: Run tests
    if (-not (Run-Tests)) {
        $script:DeploymentFailed = $true
        Show-DeploymentSummary
        exit 1
    }
    
    # Step 5: Create backup
    if (-not (Create-Backup)) {
        Write-Log "Backup failed" "WARNING"
        if (-not $Force) {
            $script:DeploymentFailed = $true
            Show-DeploymentSummary
            exit 1
        }
    }
    
    # Step 6: Deploy multi-agent system
    if (-not (Deploy-MultiAgentSystem)) {
        $script:DeploymentFailed = $true
        Show-DeploymentSummary
        exit 1
    }
    
    # Success!
    Write-Log "Deployment completed successfully" "SUCCESS"
    Show-DeploymentSummary
    exit 0
}

# Run main function
Main
