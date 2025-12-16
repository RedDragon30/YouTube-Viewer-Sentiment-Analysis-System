@echo off
echo ==========================================
echo YouTube Comments Sentiment - Setup
echo ==========================================

REM Create virtual environment
echo Creating virtual environment...
python -m venv youtube-sentiment

REM Activate virtual environment
echo Activating virtual environment...
call youtube-sentiment\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo Creating directories...
if not exist "artifacts" mkdir artifacts
if not exist "data" mkdir data

echo ==========================================
echo Setup completed successfully!
echo ==========================================
pause