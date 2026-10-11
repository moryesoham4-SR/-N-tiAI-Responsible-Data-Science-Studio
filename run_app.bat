@echo off
echo Starting NitiAI Responsible Data Science Studio...
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
streamlit run app.py
pause
