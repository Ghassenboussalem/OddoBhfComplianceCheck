@echo off
REM Multi-Agent System Rollback Script (Windows CMD)
REM This script restores the previous system state from a backup
REM Usage: rollback_multiagent.bat [backup_dir] [--force]

setlocal enabledelayedexpansion

REM Configuration
set BACKUP_DIR=%1
set FORCE_MODE=false
set PYTHON_CMD=python
set ROLLBACK_LOG=rollback_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set ROLLBACK_LOG=%ROLLBACK_LOG: =0%
set PRE_ROLLBACK_BACKUP=
set ROLLBACK_SUCCESS=true

REM Parse command line arguments
if "%2"=="--force" set FORCE_MODE=true

echo ======================================================================== >> %ROLLBACK_LOG%
echo Multi-Agent System Rollback - %date% %time% >> %ROLLBACK_LOG%
echo ======================================================================== >> %ROLLBACK_LOG%

REM Print header
echo.
echo ========================================================================
echo MULTI-AGENT SYSTEM ROLLBACK
echo ========================================================================
echo.
echo This script will restore the system from a backup
echo Press Ctrl+C to cancel at any time
echo.
echo Rollback log: %ROLLBACK_LOG%
echo.
timeout /t 2 /nobreak >nul

REM Step 1: Find backup to restore
echo ========================================================================
echo FINDING BACKUP
echo ========================================================================
echo.

if "%BACKUP_DIR%"=="" (
    REM Find latest backup
    if not exist backups (
        echo [ERROR] No backups directory found
        echo [ERROR] No backups directory found >> %ROLLBACK_LOG%
        goto :error
    )
    
    REM Find the most recent backup directory
    for /f "delims=" %%d in ('dir /b /ad /o-n backups\deployment_* backups\backup_* 2^>nul') do (
        if not defined BACKUP_DIR (
            set BACKUP_DIR=backups\%%d
        )
    )
    
    if not defined BACKUP_DIR (
        echo [ERROR] No backup directories found in backups\
        echo [ERROR] No backup directories found >> %ROLLBACK_LOG%
        goto :error
    )
    
    echo [INFO] Using latest backup: !BACKUP_DIR!
    echo [INFO] Using latest backup: !BACKUP_DIR! >> %ROLLBACK_LOG%
) else (
    REM Use specified backup directory
    if not exist "%BACKUP_DIR%" (
        echo [ERROR] Specified backup directory not found: %BACKUP_DIR%
        echo [ERROR] Backup directory not found >> %ROLLBACK_LOG%
        goto :error
    )
    echo [INFO] Using specified backup: %BACKUP_DIR%
    echo [INFO] Using specified backup: %BACKUP_DIR% >> %ROLLBACK_LOG%
)
echo.

REM Step 2: Validate backup integrity
echo ========================================================================
echo VALIDATING BACKUP INTEGRITY
echo ========================================================================
echo.

if not exist "%BACKUP_DIR%" (
    echo [ERROR] Backup directory not found: %BACKUP_DIR%
    echo [ERROR] Backup directory not found >> %ROLLBACK_LOG%
    goto :error
)

echo [SUCCESS] Backup directory exists
echo [SUCCESS] Backup directory exists >> %ROLLBACK_LOG%

REM Check for manifest file
if exist "%BACKUP_DIR%\manifest.json" (
    echo [SUCCESS] Backup manifest found
    echo [SUCCESS] Backup manifest found >> %ROLLBACK_LOG%
) else (
    echo [WARNING] No manifest file found ^(older backup format^)
    echo [WARNING] No manifest file found >> %ROLLBACK_LOG%
)

REM Check for critical files
set CRITICAL_FILES_EXIST=true
if exist "%BACKUP_DIR%\hybrid_config.json" (
    echo [SUCCESS] Found critical file: hybrid_config.json
    echo [SUCCESS] Found critical file: hybrid_config.json >> %ROLLBACK_LOG%
) else (
    echo [WARNING] Missing critical file: hybrid_config.json
    echo [WARNING] Missing critical file: hybrid_config.json >> %ROLLBACK_LOG%
    set CRITICAL_FILES_EXIST=false
)

if "%CRITICAL_FILES_EXIST%"=="false" (
    echo [WARNING] Backup may be incomplete
    echo [WARNING] Backup may be incomplete >> %ROLLBACK_LOG%
    if "%FORCE_MODE%"=="false" (
        set /p CONTINUE="Continue anyway? (yes/no): "
        if /i not "!CONTINUE!"=="yes" (
            echo [INFO] Rollback cancelled
            echo [INFO] Rollback cancelled >> %ROLLBACK_LOG%
            goto :end
        )
    )
)

echo [SUCCESS] Backup integrity check passed
echo [SUCCESS] Backup integrity check passed >> %ROLLBACK_LOG%
echo.

REM Step 3: Confirm rollback
echo ========================================================================
echo ROLLBACK CONFIRMATION
echo ========================================================================
echo.

echo [WARNING] This will restore the system from backup:
echo   Backup: %BACKUP_DIR%
echo.
echo [WARNING] Current files will be overwritten!
echo.

if "%FORCE_MODE%"=="false" (
    set /p CONFIRM="Do you want to proceed? (yes/no): "
    if /i not "!CONFIRM!"=="yes" (
        echo [INFO] Rollback cancelled by user
        echo [INFO] Rollback cancelled by user >> %ROLLBACK_LOG%
        goto :end
    )
)

echo [INFO] Rollback confirmed
echo [INFO] Rollback confirmed >> %ROLLBACK_LOG%
echo.

REM Step 4: Stop running processes
echo ========================================================================
echo STOPPING RUNNING PROCESSES
echo ========================================================================
echo.

REM Check for running Python processes
tasklist /FI "IMAGENAME eq python.exe" 2>nul | find /I "python.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo [WARNING] Python processes are running
    echo [WARNING] Python processes are running >> %ROLLBACK_LOG%
    
    if "%FORCE_MODE%"=="false" (
        set /p STOP_PROCS="Stop Python processes? (yes/no): "
        if /i "!STOP_PROCS!"=="yes" (
            taskkill /F /IM python.exe >nul 2>&1
            echo [SUCCESS] Stopped Python processes
            echo [SUCCESS] Stopped Python processes >> %ROLLBACK_LOG%
        ) else (
            echo [WARNING] Processes not stopped - rollback may fail if files are in use
            echo [WARNING] Processes not stopped >> %ROLLBACK_LOG%
        )
    )
) else (
    echo [SUCCESS] No Python processes found running
    echo [SUCCESS] No Python processes found running >> %ROLLBACK_LOG%
)
echo.

REM Step 5: Create pre-rollback backup
echo ========================================================================
echo CREATING PRE-ROLLBACK BACKUP
echo ========================================================================
echo.

set PRE_ROLLBACK_BACKUP=backups\pre_rollback_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set PRE_ROLLBACK_BACKUP=%PRE_ROLLBACK_BACKUP: =0%

if not exist backups mkdir backups
mkdir "%PRE_ROLLBACK_BACKUP%" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to create pre-rollback backup directory
    echo [ERROR] Failed to create pre-rollback backup >> %ROLLBACK_LOG%
    if "%FORCE_MODE%"=="false" (
        set /p CONTINUE="Continue without pre-rollback backup? (yes/no): "
        if /i not "!CONTINUE!"=="yes" (
            echo [INFO] Rollback cancelled
            goto :end
        )
    )
) else (
    echo [SUCCESS] Created pre-rollback backup directory: %PRE_ROLLBACK_BACKUP%
    echo [SUCCESS] Created pre-rollback backup >> %ROLLBACK_LOG%
    
    REM Backup current files
    if exist hybrid_config.json (
        copy /y hybrid_config.json "%PRE_ROLLBACK_BACKUP%\" >nul 2>&1
        echo [SUCCESS] Backed up current: hybrid_config.json
        echo [SUCCESS] Backed up: hybrid_config.json >> %ROLLBACK_LOG%
    )
    
    if exist .env (
        copy /y .env "%PRE_ROLLBACK_BACKUP%\" >nul 2>&1
        echo [SUCCESS] Backed up current: .env
        echo [SUCCESS] Backed up: .env >> %ROLLBACK_LOG%
    )
    
    if exist review_queue.json (
        copy /y review_queue.json "%PRE_ROLLBACK_BACKUP%\" >nul 2>&1
        echo [SUCCESS] Backed up current: review_queue.json
        echo [SUCCESS] Backed up: review_queue.json >> %ROLLBACK_LOG%
    )
    
    if exist checkpoints (
        xcopy /e /i /y checkpoints "%PRE_ROLLBACK_BACKUP%\checkpoints" >nul 2>&1
        echo [SUCCESS] Backed up current: checkpoints\
        echo [SUCCESS] Backed up: checkpoints\ >> %ROLLBACK_LOG%
    )
    
    if exist audit_logs (
        xcopy /e /i /y audit_logs "%PRE_ROLLBACK_BACKUP%\audit_logs" >nul 2>&1
        echo [SUCCESS] Backed up current: audit_logs\
        echo [SUCCESS] Backed up: audit_logs\ >> %ROLLBACK_LOG%
    )
    
    echo [SUCCESS] Pre-rollback backup completed
    echo [SUCCESS] Pre-rollback backup completed >> %ROLLBACK_LOG%
)
echo.

REM Step 6: Restore configuration
echo ========================================================================
echo RESTORING CONFIGURATION
echo ========================================================================
echo.

REM Restore hybrid_config.json
if exist "%BACKUP_DIR%\hybrid_config.json" (
    copy /y "%BACKUP_DIR%\hybrid_config.json" hybrid_config.json >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to restore hybrid_config.json
        echo [ERROR] Failed to restore configuration >> %ROLLBACK_LOG%
        goto :error
    )
    echo [SUCCESS] Restored: hybrid_config.json
    echo [SUCCESS] Restored: hybrid_config.json >> %ROLLBACK_LOG%
    
    REM Disable multi-agent mode
    %PYTHON_CMD% -c "import json; config = json.load(open('hybrid_config.json')); config.setdefault('multi_agent', {})['enabled'] = False; json.dump(config, open('hybrid_config.json', 'w'), indent=2)" >> %ROLLBACK_LOG% 2>&1
    if errorlevel 1 (
        echo [WARNING] Failed to disable multi-agent mode
        echo [WARNING] Failed to disable multi-agent mode >> %ROLLBACK_LOG%
    ) else (
        echo [SUCCESS] Disabled multi-agent mode in configuration
        echo [SUCCESS] Disabled multi-agent mode >> %ROLLBACK_LOG%
    )
) else (
    echo [WARNING] No configuration file in backup
    echo [WARNING] No configuration file in backup >> %ROLLBACK_LOG%
)

REM Restore .env
if exist "%BACKUP_DIR%\.env" (
    copy /y "%BACKUP_DIR%\.env" .env >nul 2>&1
    echo [SUCCESS] Restored: .env
    echo [SUCCESS] Restored: .env >> %ROLLBACK_LOG%
)

REM Restore review queue
if exist "%BACKUP_DIR%\review_queue.json" (
    copy /y "%BACKUP_DIR%\review_queue.json" review_queue.json >nul 2>&1
    echo [SUCCESS] Restored: review_queue.json
    echo [SUCCESS] Restored: review_queue.json >> %ROLLBACK_LOG%
)

REM Restore audit logs
if exist "%BACKUP_DIR%\audit_logs" (
    if exist audit_logs rmdir /s /q audit_logs >nul 2>&1
    xcopy /e /i /y "%BACKUP_DIR%\audit_logs" audit_logs >nul 2>&1
    echo [SUCCESS] Restored: audit_logs\
    echo [SUCCESS] Restored: audit_logs\ >> %ROLLBACK_LOG%
)

REM Restore checkpoints
if exist "%BACKUP_DIR%\checkpoints" (
    if exist checkpoints rmdir /s /q checkpoints >nul 2>&1
    xcopy /e /i /y "%BACKUP_DIR%\checkpoints" checkpoints >nul 2>&1
    echo [SUCCESS] Restored: checkpoints\
    echo [SUCCESS] Restored: checkpoints\ >> %ROLLBACK_LOG%
)

echo [SUCCESS] Configuration restoration complete
echo [SUCCESS] Configuration restoration complete >> %ROLLBACK_LOG%
echo.

REM Step 7: Restore files
echo ========================================================================
echo RESTORING FILES FROM BACKUP
echo ========================================================================
echo.

echo [INFO] Restoring files from backup...
echo [INFO] Restoring files >> %ROLLBACK_LOG%

REM Copy all files from backup (excluding manifest.json)
for /r "%BACKUP_DIR%" %%f in (*) do (
    if not "%%~nxf"=="manifest.json" (
        set "source=%%f"
        set "relative=!source:%BACKUP_DIR%\=!"
        
        REM Create directory if needed
        for %%d in ("!relative!") do set "target_dir=%%~dpd"
        if not "!target_dir!"=="" (
            if not exist "!target_dir!" mkdir "!target_dir!" >nul 2>&1
        )
        
        REM Copy file
        copy /y "%%f" "!relative!" >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Failed to restore: !relative!
            echo [ERROR] Failed to restore: !relative! >> %ROLLBACK_LOG%
        ) else (
            echo [SUCCESS] Restored: !relative!
            echo [SUCCESS] Restored: !relative! >> %ROLLBACK_LOG%
        )
    )
)

echo [SUCCESS] File restoration complete
echo [SUCCESS] File restoration complete >> %ROLLBACK_LOG%
echo.

REM Step 8: Verify rollback success
echo ========================================================================
echo VERIFYING ROLLBACK SUCCESS
echo ========================================================================
echo.

REM Check critical files exist
set VERIFICATION_PASSED=true

if exist check.py (
    echo [SUCCESS] Verified: check.py exists
    echo [SUCCESS] Verified: check.py >> %ROLLBACK_LOG%
) else (
    echo [ERROR] Missing: check.py
    echo [ERROR] Missing: check.py >> %ROLLBACK_LOG%
    set VERIFICATION_PASSED=false
)

if exist hybrid_config.json (
    echo [SUCCESS] Verified: hybrid_config.json exists
    echo [SUCCESS] Verified: hybrid_config.json >> %ROLLBACK_LOG%
    
    REM Verify configuration is valid JSON
    %PYTHON_CMD% -c "import json; json.load(open('hybrid_config.json'))" >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Configuration file is invalid
        echo [ERROR] Configuration file is invalid >> %ROLLBACK_LOG%
        set VERIFICATION_PASSED=false
    ) else (
        echo [SUCCESS] Configuration file is valid JSON
        echo [SUCCESS] Configuration is valid >> %ROLLBACK_LOG%
    )
) else (
    echo [ERROR] Missing: hybrid_config.json
    echo [ERROR] Missing: hybrid_config.json >> %ROLLBACK_LOG%
    set VERIFICATION_PASSED=false
)

REM Test basic functionality
if exist check.py (
    echo [INFO] Testing basic system functionality...
    echo [INFO] Testing system >> %ROLLBACK_LOG%
    %PYTHON_CMD% check.py --help >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Basic system test failed
        echo [WARNING] System test failed >> %ROLLBACK_LOG%
    ) else (
        echo [SUCCESS] Basic system test passed
        echo [SUCCESS] System test passed >> %ROLLBACK_LOG%
    )
)

if "%VERIFICATION_PASSED%"=="false" (
    echo [ERROR] Rollback verification failed
    echo [ERROR] Rollback verification failed >> %ROLLBACK_LOG%
    goto :error
)

echo [SUCCESS] Rollback verification complete
echo [SUCCESS] Rollback verification complete >> %ROLLBACK_LOG%
echo.

REM Success!
echo ========================================================================
echo ROLLBACK SUMMARY
echo ========================================================================
echo.
echo [SUCCESS] Rollback completed successfully!
echo [SUCCESS] Rollback completed >> %ROLLBACK_LOG%
echo.
echo [INFO] Rollback Log: %ROLLBACK_LOG%
echo [INFO] Restored From: %BACKUP_DIR%
if not "%PRE_ROLLBACK_BACKUP%"=="" (
    echo [INFO] Pre-Rollback Backup: %PRE_ROLLBACK_BACKUP%
    echo   ^(Use this to restore if rollback was incorrect^)
)
echo.
echo System has been restored to previous state
echo.
echo Next Steps:
echo   1. Review the rollback log for any warnings
echo   2. Test the system: python check.py exemple.json
echo   3. Verify your configuration in hybrid_config.json
echo   4. Check that all expected files are present
echo.
echo If you need to restore the multi-agent system:
echo   deploy_multiagent.bat
echo.
echo ========================================================================
goto :end

:error
echo.
echo ========================================================================
echo [ERROR] ROLLBACK FAILED
echo ========================================================================
echo.
echo Please check the rollback log for details: %ROLLBACK_LOG%
echo.
echo Recovery Options:
echo   1. Review the rollback log: %ROLLBACK_LOG%
echo   2. Try rollback again with --force flag
echo   3. Manually restore files from: %BACKUP_DIR%
if not "%PRE_ROLLBACK_BACKUP%"=="" (
    echo   4. Restore pre-rollback state from: %PRE_ROLLBACK_BACKUP%
)
echo.
echo ========================================================================
echo [ERROR] Rollback failed >> %ROLLBACK_LOG%
exit /b 1

:end
endlocal
exit /b 0
