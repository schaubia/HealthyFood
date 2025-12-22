@echo off
REM Food Health Analyzer - Windows Setup Script

echo.
echo 🍎 Food Health Analyzer - Setup Script
echo ========================================
echo.

REM Check Python installation
echo ✓ Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo ✓ Virtual environment created
)
echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✓ pip upgraded
echo.

REM Install requirements
echo 📥 Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt --quiet
echo ✓ Dependencies installed
echo.

REM Check USDA API Key
echo 🔑 Checking USDA API Key...
if "%USDA_API_KEY%"=="" (
    echo ⚠️  USDA_API_KEY not set. Using DEMO_KEY ^(limited to 1000 requests/hour^)
    echo.
    echo To get better rate limits:
    echo 1. Sign up at: https://fdc.nal.usda.gov/api-key-signup.html
    echo 2. Set your key: set USDA_API_KEY=your_key_here
    echo.
) else (
    echo ✓ USDA_API_KEY is set
)
echo.

REM Create directories
echo 📁 Creating directories...
if not exist examples mkdir examples
if not exist models mkdir models
echo ✓ Directories created
echo.

REM Final message
echo ✅ Setup complete!
echo.
echo To run the application:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate
echo   2. Run the app:
echo      python app.py
echo.
echo The app will open at: http://127.0.0.1:7860
echo.
echo Happy analyzing! 🎉
echo.
pause
