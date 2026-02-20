@echo off
echo ========================================
echo  Push Virtual Try-On to GitHub
echo ========================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in PATH
    echo Please install Git from: https://git-scm.com/download/win
    echo Or use Git Bash terminal instead
    pause
    exit /b 1
)

echo Step 1: Adding all files...
git add .

echo.
echo Step 2: Creating commit...
git commit -m "Initial commit: Virtual Try-On Application"

echo.
echo Step 3: Checking remote...
git remote -v

echo.
echo ========================================
echo Next steps:
echo ========================================
echo.
echo 1. Create a new repository on GitHub:
echo    - Go to https://github.com/new
echo    - Name it: virtual-tryon
echo    - DO NOT initialize with README
echo    - Click "Create repository"
echo.
echo 2. Then run these commands in Git Bash:
echo    git remote add origin https://github.com/YOUR_USERNAME/virtual-tryon.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo ========================================
pause

@REM Made with Bob
