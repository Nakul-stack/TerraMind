@echo off
echo ═══════════════════════════════════════
echo   TerraMind — Training All Models
echo ═══════════════════════════════════════
cd /d "%~dp0\.."
python -m backend.models.train_crop_recommender
python -m backend.models.train_yield_predictor
python -m backend.models.train_agri_advisor
echo.
echo ═══ All models trained! ═══
pause
