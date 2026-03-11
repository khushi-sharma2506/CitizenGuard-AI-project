# 🛡️ CitizenGuard AI — Location-Based Hazard Alert System

An AI-powered civic safety platform that detects urban hazards, classifies severity using Machine Learning, and simulates real-time SMS alerts to nearby citizens.

> **Inspired by real accidents in Delhi-Noida** where people died due to unmarked construction pits and potholes.

🔗 **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)  
📁 **GitHub:** [github.com/khushi-sharma2506/citizenguard-ai](https://github.com/khushi-sharma2506/citizenguard-ai)

---

## 📸 Features

| Page | Description |
|---|---|
| 🏠 Dashboard | Live hazard stats, severity charts, recent alerts |
| 📍 Report Hazard | Citizens submit hazards with location + AI prediction |
| 🗺️ Hazard Map | Interactive map with color-coded severity pins |
| 🤖 AI Classifier | Gradient Boosting model predicts Low/Medium/High/Critical |
| 📊 Admin Panel | Manage reports, broadcast SMS, view model metrics |

---

## 🧠 AI/ML Component

- **Algorithm:** Gradient Boosting Classifier
- **Features:** Hazard type, time, reports count, proximity to hospital/school, weather, population density
- **Output:** Severity level (Low / Medium / High / Critical) + confidence score
- **Accuracy:** 90%+
- **Training data:** 3,000 synthetic hazard samples with rule-based labeling

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | Gradient Boosting ML model |
| Streamlit | Interactive web dashboard |
| Plotly | Maps and visualizations |
| Pandas / NumPy | Data processing |
| Streamlit Cloud | Free deployment |

---

## 🚀 Run Locally

```bash
git clone https://github.com/khushi-sharma2506/citizenguard-ai.git
cd citizenguard-ai
pip install -r requirements.txt
python train_model.py   # trains AI model
streamlit run app.py    # launches dashboard
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → `app.py` → Deploy
4. App auto-trains model on first launch ✅

---

## 📝 Resume Description

> *"Built CitizenGuard AI, a location-based hazard alert system using Gradient Boosting ML classifier to predict hazard severity (Low/Medium/High/Critical) with 90%+ accuracy. Features 5-page Streamlit dashboard with interactive hazard map, citizen reporting, SMS alert simulation, and admin panel — inspired by real Delhi-Noida infrastructure accidents."*

---

## 👤 Author

**Khushi Sharma** — B.Tech CSE (AI & ML), Graphic Era University, Dehradun

- GitHub: [@khushi-sharma2506](https://github.com/khushi-sharma2506)
