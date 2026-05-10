---
title: Student Productivity Score
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---
# StudyIQ — Student Productivity Predictor Web App

## Setup (3 steps)

1. Install dependencies:
   pip install flask scikit-learn pandas numpy

2. Run the server:
   python app.py

3. Open your browser at:
   http://localhost:5000

## Project Structure
  app.py              — Flask backend (API + routes)
  model_bundle.pkl    — Trained Random Forest model (R² = 0.98)
  templates/
    index.html        — Full frontend (HTML + CSS + JS)

## API
POST /predict
  Body: JSON with all 16 student features
  Returns: score, label (Low/Medium/High), tips, feature importances

## Model Details
  Algorithm : Random Forest Regressor (300 trees)
  Dataset   : 20,000 students, 18 features
  R² Score  : 0.9826
  Target    : productivity_score (0–100, continuous)
  Classes   : Low (≤42.6) | Medium (≤57.9) | High (>57.9)
