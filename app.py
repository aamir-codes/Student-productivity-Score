from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle, os

app = Flask(__name__)

# Load model bundle
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), 'model_bundle.pkl')
with open(BUNDLE_PATH, 'rb') as f:
    bundle = pickle.load(f)

regressor    = bundle['regressor']
scaler       = bundle['scaler']
TOP_FEATURES = bundle['top_features']
LOW_THRESH   = bundle['low_thresh']
HIGH_THRESH  = bundle['high_thresh']

def engineer_features(d):
    d['screen_time_total'] = (d.get('phone_usage_hours', 0) +
                               d.get('social_media_hours', 0) +
                               d.get('youtube_hours', 0) +
                               d.get('gaming_hours', 0))
    d['study_sleep_ratio'] = d.get('study_hours_per_day', 0) / (d.get('sleep_hours', 1) + 1e-5)
    d['wellness_score']    = (d.get('exercise_minutes', 0) / 60
                               - d.get('stress_level', 5)
                               + d.get('sleep_hours', 7))
    return d

def classify(score):
    if score <= LOW_THRESH:   return 'Low',    0
    elif score <= HIGH_THRESH: return 'Medium', 1
    else:                      return 'High',   2

def get_tips(data, score):
    tips = []
    if data.get('study_hours_per_day', 0) < 3:
        tips.append("📚 Increase daily study time — even 30 more minutes can significantly boost productivity.")
    if data.get('sleep_hours', 0) < 6:
        tips.append("😴 You're sleep-deprived. Aim for 7–8 hours; sleep is directly tied to focus and retention.")
    screen = (data.get('phone_usage_hours',0) + data.get('social_media_hours',0) +
              data.get('youtube_hours',0) + data.get('gaming_hours',0))
    if screen > 6:
        tips.append(f"📱 Your total screen time is {screen:.1f} hrs/day. Try cutting it by 2 hours using app timers.")
    if data.get('stress_level', 0) >= 7:
        tips.append("🧘 High stress detected. Consider mindfulness, exercise, or speaking to a counselor.")
    if data.get('exercise_minutes', 0) < 20:
        tips.append("🏃 Just 20–30 min of daily exercise measurably improves focus and academic performance.")
    if data.get('attendance_percentage', 100) < 70:
        tips.append("🏫 Low attendance is hurting your score. Regular class presence strongly correlates with productivity.")
    if data.get('focus_score', 0) < 50:
        tips.append("🎯 Try the Pomodoro technique: 25-min focus sessions with 5-min breaks to build concentration.")
    if not tips:
        tips.append("🌟 Excellent habits! Keep up your study routine, sleep schedule, and low screen time.")
    return tips[:3]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        d = {
            'age'                  : float(data.get('age', 20)),
            'gender'               : 1 if data.get('gender','').lower() == 'male' else 0,
            'study_hours_per_day'  : float(data.get('study_hours_per_day', 5)),
            'sleep_hours'          : float(data.get('sleep_hours', 7)),
            'phone_usage_hours'    : float(data.get('phone_usage_hours', 3)),
            'social_media_hours'   : float(data.get('social_media_hours', 2)),
            'youtube_hours'        : float(data.get('youtube_hours', 1.5)),
            'gaming_hours'         : float(data.get('gaming_hours', 1)),
            'breaks_per_day'       : float(data.get('breaks_per_day', 5)),
            'coffee_intake_mg'     : float(data.get('coffee_intake_mg', 150)),
            'exercise_minutes'     : float(data.get('exercise_minutes', 30)),
            'assignments_completed': float(data.get('assignments_completed', 8)),
            'attendance_percentage': float(data.get('attendance_percentage', 80)),
            'stress_level'         : float(data.get('stress_level', 5)),
            'focus_score'          : float(data.get('focus_score', 60)),
            'final_grade'          : float(data.get('final_grade', 70)),
        }
        d = engineer_features(d)
        row    = pd.DataFrame([{f: d.get(f, 0) for f in TOP_FEATURES}])
        row_sc = pd.DataFrame(scaler.transform(row), columns=TOP_FEATURES)
        score  = float(regressor.predict(row_sc)[0])
        score  = max(0.0, min(100.0, score))
        label, level = classify(score)

        # Feature importance for radar chart
        importances = dict(zip(TOP_FEATURES, regressor.feature_importances_))

        return jsonify({
            'success'      : True,
            'score'        : round(score, 1),
            'label'        : label,
            'level'        : level,
            'low_thresh'   : round(LOW_THRESH, 1),
            'high_thresh'  : round(HIGH_THRESH, 1),
            'tips'         : get_tips(d, score),
            'screen_time'  : round(d['screen_time_total'], 1),
            'wellness'     : round(d['wellness_score'], 2),
            'study_ratio'  : round(d['study_sleep_ratio'], 2),
            'top_features' : TOP_FEATURES[:6],
            'importances'  : {k: round(v * 100, 1) for k, v in list(importances.items())[:6]},
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
