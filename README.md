# meta-ml-titanic-leaderboard

run_assessment.py (localhost)
    ↓ POST
Green Agent (localhost:9010)
    ↓ Downloads CSVs from GitHub
    ↓ Creates TRAINING_DATA:{json}
    ↓ POST via A2A
Solver (http://solver:9009 - Docker network)
    ↓ Parses CSV from JSON
    ↓ Trains XGBoost model
    ↓ Returns predictions + research
    ↑ Artifacts
Green Agent
    ↓ Evaluates accuracy/F1
    ↓ Returns results
    ↑ 
run_assessment.py
    ↓ Commits to GitHub
