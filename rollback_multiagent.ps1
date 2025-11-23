#!/usr/bin/env pwsh
# Multi-Agent System Rollback Script
# This script restores the previous system state from a backup
# Usage: .\rollback_multiagent.ps1 [-BackupDir <path>] [-Force]

param(
    [string]$BackupDir = "",
    [switch]$Force = $false,
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
$script:RollbackFailed = $false
$script:RollbackLog = "rollback_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Add-Content -Path $script:RollbackLog -Value $logMessage
    
    switch ($Level) {
        "ERROR" { Write-Error $Message }
        "WARNING" { Write-Warning $Message }
        "SUCCESS" { Write-Success $Message }
        default { Write-Info $Message }
    }
}

function Find-LatestBackup {
    Write-Header "FINDING LATEST BACKUP"
    
    if (-not (Test-Path "backups")) {
        Write-Log "No backups directory found" "ERROR"
        return $null
    }
    
    # Find all backup directories
    $backupDirs = Get-ChildItem -Path "backups" -Directory | 
                  Where-Object { $_.Name -match "^(deployment_|backup_)\d{8}_\d{6}$" } |
                  Sort-Object Name -Descending
    
    if ($backupDirs.Count -eq 0) {
        Write-Log "No backup directories found in backups/" "ERROR"
        return $null
    }
    
    Write-Log "Found $($backupDirs.Count) backup(s)" "INFO"
    
    # Display available backups
    Write-Host ""
    Write-Host "Available backups:" -ForegroundColor Cyan
    for ($i = 0; $i -lt [Math]::Min(5, $backupDirs.Count); $i++) {
        $backup = $backupDirs[$i]
        $timestamp = $backup.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "  $($i+1). $($backup.Name) (Created: $timestamp)" -ForegroundColor White
    }
    Write-Host ""
    
    # Return the latest backup
    $latestBackup = $backupDirs[0].FullName
    Write-Log "Latest backup: $latestBackup" "SUCCESS"
    return $latestBackup
}

function Confirm-Rollback {
    param($BackupPath)
    
    Write-Header "ROLLBACK CONFIRMATION"
    
    Write-Host ""
    Write-Warning "This will restore the system from backup:"
    Write-Host "  Backup: $BackupPath" -ForegroundColor White
    Write-Host ""
    Write-Warning "Current files will be overwritten!"
    Write-Host ""
    
    if ($Force) {
        Write-Log "Rollback confirmed (forced)" "INFO"
        return $true
    }
    
    $response = Read-Host "Do you want to proceed? (yes/no)"
    
    if ($response -eq "yes") {
        Write-Log "Rollback confirmed by user" "INFO"
        return $true
    } else {
        Write-Log "Rollback cancelled by user" "WARNING"
        return $false
    }
}

function Test-BackupIntegrity {
    param($BackupPath)
    
    Write-Header "VALIDATING BACKUP INTEGRITY"
    
    # Check if backup directory exists
    if (-not (Test-Path $BackupPath)) {
        Write-Log "Backup directory not found: $BackupPath" "ERROR"
        return $false
    }
    
    Write-Log "Backup directory exists" "SUCCESS"
    
    # Check for manifest file
    $manifestPath = Join-Path $BackupPath "manifest.json"
    if (Test-Path $manifestPath) {
        try {
            $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
            Write-Log "Backup manifest found" "SUCCESS"
            Write-Log "  Backup timestamp: $($manifest.timestamp)" "INFO"
            Write-Log "  Python version: $($manifest.python_version)" "INFO"
            Write-Log "  Files: $($manifest.files.Count)" "INFO"
            Write-Log "  Directories: $($manifest.directories.Count)" "INFO"
        } catch {
            Write-Log "Failed to parse manifest: $_" "WARNING"
        }
    } else {
        Write-Log "No manifest file found (older backup format)" "WARNING"
    }
    
    # Check for critical files
    $criticalFiles = @(
        "hybrid_config.json"
    )
    
    $allCriticalFilesExist = $true
    foreach ($file in $criticalFiles) {
        $filePath = Join-Path $BackupPath $file
        if (Test-Path $filePath) {
            Write-Log "Found critical file: $file" "SUCCESS"
        } else {
            Write-Log "Missing critical file: $file" "WARNING"
            $allCriticalFilesExist = $false
        }
    }
    
    if (-not $allCriticalFilesExist) {
        Write-Log "Backup may be incomplete" "WARNING"
        if (-not $Force) {
            Write-Host ""
            $response = Read-Host "Continue anyway? (yes/no)"
            if ($response -ne "yes") {
                return $false
            }
        }
    }
    
    Write-Log "Backup integrity check passed" "SUCCESS"
    return $true
}

function Stop-RunningProcesses {
    Write-Header "STOPPING RUNNING PROCESSES"
    
    # Check for running Python processes that might be using the system
    $pythonProcesses = Get-Process -Name "python*" -ErrorAction SilentlyContinue
    
    if ($pythonProcesses) {
        Write-Log "Found $($pythonProcesses.Count) Python process(es) running" "WARNING"
        Write-Host ""
        Write-Warning "The following Python processes are running:"
        foreach ($proc in $pythonProcesses) {
            Write-Host "  PID: $($proc.Id) - $($proc.ProcessName)" -ForegroundColor White
        }
        Write-Host ""
        
        if (-not $Force) {
            $response = Read-Host "Stop these processes? (yes/no)"
            if ($response -eq "yes") {
                foreach ($proc in $pythonProcesses) {
                    try {
                        Stop-Process -Id $proc.Id -Force
                        Write-Log "Stopped process: $($proc.Id)" "SUCCESS"
                    } catch {
                        Write-Log "Failed to stop process $($proc.Id): $_" "WARNING"
                    }
                }
            } else {
                Write-Log "Processes not stopped - rollback may fail if files are in use" "WARNING"
            }
        }
    } else {
        Write-Log "No Python processes found running" "SUCCESS"
    }
}

function Create-PreRollbackBackup {
    Write-Header "CREATING PRE-ROLLBACK BACKUP"
    
    $preRollbackDir = "backups/pre_rollback_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    try {
        New-Item -ItemType Directory -Path $preRollbackDir -Force | Out-Null
        Write-Log "Created pre-rollback backup directory: $preRollbackDir" "SUCCESS"
    } catch {
        Write-Log "Failed to create pre-rollback backup: $_" "ERROR"
        return $null
    }
    
    # Backup current state
    $filesToBackup = @(
        "hybrid_config.json",
        ".env",
        "review_queue.json"
    )
    
    $dirsToBackup = @(
        "checkpoints",
        "audit_logs"
    )
    
    # Backup files
    foreach ($file in $filesToBackup) {
        if (Test-Path $file) {
            try {
                Copy-Item $file -Destination $preRollbackDir -Force
                Write-Log "Backed up current: $file" "SUCCESS"
            } catch {
                Write-Log "Failed to backup ${file}: $_" "WARNING"
            }
        }
    }
    
    # Backup directories
    foreach ($dir in $dirsToBackup) {
        if (Test-Path $dir) {
            try {
                Copy-Item $dir -Destination $preRollbackDir -Recurse -Force
                Write-Log "Backed up current: $dir/" "SUCCESS"
            } catch {
                Write-Log "Failed to backup ${dir}: $_" "WARNING"
            }
        }
    }
    
    Write-Log "Pre-rollback backup completed: $preRollbackDir" "SUCCESS"
    return $preRollbackDir
}

function Restore-Files {
    param($BackupPath)
    
    Write-Header "RESTORING FILES FROM BACKUP"
    
    # Get all files from backup
    $backupFiles = Get-ChildItem -Path $BackupPath -File -Recurse
    
    if ($backupFiles.Count -eq 0) {
        Write-Log "No files found in backup" "WARNING"
        return $false
    }
    
    Write-Log "Found $($backupFiles.Count) file(s) to restore" "INFO"
    
    $restoredCount = 0
    $failedCount = 0
    
    foreach ($file in $backupFiles) {
        # Get relative path
        $relativePath = $file.FullName.Substring($BackupPath.Length + 1)
        
        # Skip manifest file
        if ($relativePath -eq "manifest.json") {
            continue
        }
        
        try {
            # Create directory if needed
            $targetDir = Split-Path $relativePath -Parent
            if ($targetDir -and -not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            
            # Copy file
            Copy-Item $file.FullName -Destination $relativePath -Force
            Write-Log "Restored: $relativePath" "SUCCESS"
            $restoredCount++
        } catch {
            Write-Log "Failed to restore ${relativePath}: $_" "ERROR"
            $failedCount++
        }
    }
    
    Write-Host ""
    Write-Log "Restoration complete: $restoredCount succeeded, $failedCount failed" "INFO"
    
    if ($failedCount -gt 0) {
        Write-Log "Some files failed to restore" "WARNING"
        return $false
    }
    
    return $true
}

function Restore-Configuration {
    param($BackupPath)
    
    Write-Header "RESTORING CONFIGURATION"
    
    # Restore hybrid_config.json
    $configPath = Join-Path $BackupPath "hybrid_config.json"
    if (Test-Path $configPath) {
        try {
            Copy-Item $configPath -Destination "hybrid_config.json" -Force
            Write-Log "Restored: hybrid_config.json" "SUCCESS"
            
            # Verify configuration
            $config = Get-Content "hybrid_config.json" -Raw | ConvertFrom-Json
            
            # Disable multi-agent mode
            if ($config.multi_agent) {
                $config.multi_agent.enabled = $false
                $config | ConvertTo-Json -Depth 10 | Out-File "hybrid_config.json" -Encoding UTF8
                Write-Log "Disabled multi-agent mode in configuration" "SUCCESS"
            }
        } catch {
            Write-Log "Failed to restore configuration: $_" "ERROR"
            return $false
        }
    } else {
        Write-Log "No configuration file in backup" "WARNING"
    }
    
    # Restore .env
    $envPath = Join-Path $BackupPath ".env"
    if (Test-Path $envPath) {
        try {
            Copy-Item $envPath -Destination ".env" -Force
            Write-Log "Restored: .env" "SUCCESS"
        } catch {
            Write-Log "Failed to restore .env: $_" "WARNING"
        }
    }
    
    # Restore review queue
    $queuePath = Join-Path $BackupPath "review_queue.json"
    if (Test-Path $queuePath) {
        try {
            Copy-Item $queuePath -Destination "review_queue.json" -Force
            Write-Log "Restored: review_queue.json" "SUCCESS"
        } catch {
            Write-Log "Failed to restore review queue: $_" "WARNING"
        }
    }
    
    # Restore audit logs
    $auditPath = Join-Path $BackupPath "audit_logs"
    if (Test-Path $auditPath) {
        try {
            if (Test-Path "audit_logs") {
                Remove-Item "audit_logs" -Recurse -Force
            }
            Copy-Item $auditPath -Destination "audit_logs" -Recurse -Force
            Write-Log "Restored: audit_logs/" "SUCCESS"
        } catch {
            Write-Log "Failed to restore audit logs: $_" "WARNING"
        }
    }
    
    # Restore checkpoints
    $checkpointPath = Join-Path $BackupPath "checkpoints"
    if (Test-Path $checkpointPath) {
        try {
            if (Test-Path "checkpoints") {
                Remove-Item "checkpoints" -Recurse -Force
            }
            Copy-Item $checkpointPath -Destination "checkpoints" -Recurse -Force
            Write-Log "Restored: checkpoints/" "SUCCESS"
        } catch {
            Write-Log "Failed to restore checkpoints: $_" "WARNING"
        }
    }
    
    Write-Log "Configuration restoration complete" "SUCCESS"
    return $true
}

function Test-RollbackSuccess {
    Write-Header "VERIFYING ROLLBACK SUCCESS"
    
    # Check critical files exist
    $criticalFiles = @(
        "check.py",
        "hybrid_config.json"
    )
    
    $allFilesExist = $true
    foreach ($file in $criticalFiles) {
        if (Test-Path $file) {
            Write-Log "Verified: $file exists" "SUCCESS"
        } else {
            Write-Log "Missing: $file" "ERROR"
            $allFilesExist = $false
        }
    }
    
    if (-not $allFilesExist) {
        Write-Log "Rollback verification failed - missing critical files" "ERROR"
        return $false
    }
    
    # Verify configuration
    if (Test-Path "hybrid_config.json") {
        try {
            $config = Get-Content "hybrid_config.json" -Raw | ConvertFrom-Json
            Write-Log "Configuration file is valid JSON" "SUCCESS"
            
            # Check multi-agent mode is disabled
            if ($config.multi_agent -and $config.multi_agent.enabled -eq $false) {
                Write-Log "Multi-agent mode is disabled" "SUCCESS"
            } elseif (-not $config.multi_agent) {
                Write-Log "Multi-agent configuration not present (pre-migration state)" "SUCCESS"
            } else {
                Write-Log "Multi-agent mode is still enabled" "WARNING"
            }
        } catch {
            Write-Log "Configuration file is invalid: $_" "ERROR"
            return $false
        }
    }
    
    # Test basic functionality
    if (Test-Path "check.py") {
        Write-Log "Testing basic system functionality..." "INFO"
        try {
            & $PythonCmd check.py --help | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Log "Basic system test passed" "SUCCESS"
            } else {
                Write-Log "Basic system test failed" "WARNING"
            }
        } catch {
            Write-Log "Error testing system: $_" "WARNING"
        }
    }
    
    Write-Log "Rollback verification complete" "SUCCESS"
    return $true
}

function Show-RollbackSummary {
    param($BackupPath, $PreRollbackBackup)
    
    Write-Header "ROLLBACK SUMMARY"
    
    Write-Host ""
    Write-Host "Rollback Status:" -ForegroundColor Cyan
    if ($script:RollbackFailed) {
        Write-Host "  ✗ FAILED" -ForegroundColor Red
    } else {
        Write-Host "  ✓ SUCCESS" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Rollback Log: $script:RollbackLog" -ForegroundColor Cyan
    Write-Host "Restored From: $BackupPath" -ForegroundColor Cyan
    
    if ($PreRollbackBackup) {
        Write-Host "Pre-Rollback Backup: $PreRollbackBackup" -ForegroundColor Cyan
        Write-Host "  (Use this to restore if rollback was incorrect)" -ForegroundColor Yellow
    }
    
    if (-not $script:RollbackFailed) {
        Write-Host ""
        Write-Host "System has been restored to previous state" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Yellow
        Write-Host "  1. Review the rollback log for any warnings" -ForegroundColor White
        Write-Host "  2. Test the system: python check.py exemple.json" -ForegroundColor White
        Write-Host "  3. Verify your configuration in hybrid_config.json" -ForegroundColor White
        Write-Host "  4. Check that all expected files are present" -ForegroundColor White
        Write-Host ""
        Write-Host "If you need to restore the multi-agent system:" -ForegroundColor Yellow
        Write-Host "  .\deploy_multiagent.ps1" -ForegroundColor White
    } else {
        Write-Host ""
        Write-Host "Rollback failed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Recovery Options:" -ForegroundColor Yellow
        Write-Host "  1. Review the rollback log: $script:RollbackLog" -ForegroundColor White
        Write-Host "  2. Try rollback again with -Force flag" -ForegroundColor White
        Write-Host "  3. Manually restore files from: $BackupPath" -ForegroundColor White
        if ($PreRollbackBackup) {
            Write-Host "  4. Restore pre-rollback state from: $PreRollbackBackup" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "$('='*70)" -ForegroundColor Yellow
}

# Main rollback flow
function Main {
    Write-Header "MULTI-AGENT SYSTEM ROLLBACK"
    Write-Log "Starting rollback..." "INFO"
    Write-Log "Rollback log: $script:RollbackLog" "INFO"
    
    # Step 1: Find backup to restore
    if (-not $BackupDir) {
        $BackupDir = Find-LatestBackup
        if (-not $BackupDir) {
            Write-Log "No backup found to restore" "ERROR"
            $script:RollbackFailed = $true
            Show-RollbackSummary -BackupPath "N/A" -PreRollbackBackup $null
            exit 1
        }
    } else {
        # Use specified backup directory
        if (-not (Test-Path $BackupDir)) {
            Write-Log "Specified backup directory not found: $BackupDir" "ERROR"
            $script:RollbackFailed = $true
            Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $null
            exit 1
        }
        Write-Log "Using specified backup: $BackupDir" "INFO"
    }
    
    # Step 2: Validate backup integrity
    if (-not (Test-BackupIntegrity -BackupPath $BackupDir)) {
        Write-Log "Backup integrity check failed" "ERROR"
        $script:RollbackFailed = $true
        Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $null
        exit 1
    }
    
    # Step 3: Confirm rollback
    if (-not (Confirm-Rollback -BackupPath $BackupDir)) {
        Write-Log "Rollback cancelled" "INFO"
        exit 0
    }
    
    # Step 4: Stop running processes
    Stop-RunningProcesses
    
    # Step 5: Create pre-rollback backup
    $preRollbackBackup = Create-PreRollbackBackup
    if (-not $preRollbackBackup) {
        Write-Log "Failed to create pre-rollback backup" "WARNING"
        if (-not $Force) {
            $response = Read-Host "Continue without pre-rollback backup? (yes/no)"
            if ($response -ne "yes") {
                Write-Log "Rollback cancelled" "INFO"
                exit 0
            }
        }
    }
    
    # Step 6: Restore configuration
    if (-not (Restore-Configuration -BackupPath $BackupDir)) {
        Write-Log "Configuration restoration failed" "ERROR"
        $script:RollbackFailed = $true
        Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $preRollbackBackup
        exit 1
    }
    
    # Step 7: Restore files
    if (-not (Restore-Files -BackupPath $BackupDir)) {
        Write-Log "File restoration had errors" "WARNING"
        if (-not $Force) {
            $script:RollbackFailed = $true
            Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $preRollbackBackup
            exit 1
        }
    }
    
    # Step 8: Verify rollback success
    if (-not (Test-RollbackSuccess)) {
        Write-Log "Rollback verification failed" "ERROR"
        $script:RollbackFailed = $true
        Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $preRollbackBackup
        exit 1
    }
    
    # Success!
    Write-Log "Rollback completed successfully" "SUCCESS"
    Show-RollbackSummary -BackupPath $BackupDir -PreRollbackBackup $preRollbackBackup
    exit 0
}

# Run main function
Main
