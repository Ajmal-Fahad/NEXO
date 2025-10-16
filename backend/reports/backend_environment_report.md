# 🧱 NEXO Backend Environment Report

**Generated On:** 13-Oct-2025 | 05:15 AM IST  
**Environment Path:** /Users/ajmalfahad/NEXO/backend/.venv  
**Python Version:** 3.13.7 (Homebrew Installation)

---

## ✅ Core Environment Details

| Category | Tool / Library | Version | Purpose |
|-----------|----------------|----------|----------|
| Core Framework | FastAPI | 0.119.0 | Main backend API framework |
| ASGI Server | Uvicorn | 0.37.0 | Runs FastAPI app |
| Data Processing | Pandas | 2.3.3 | CSV and financial data handling |
| AI & NLP | OpenAI | 2.3.0 | LLM-powered PDF & sentiment analysis |
| Image Processing | Pillow | 11.3.0 | Logo and banner handling |
| PDF Parsing | PyPDF | 6.1.1 | Reads & extracts PDF text |
| Automation | Watchdog | 6.0.0 | Monitors PDF folder for new files |
| Env Management | Python-dotenv | 1.1.1 | Loads .env configuration |
| HTTP Client | HTTPX | 0.28.1 | Async API calls (used by OpenAI) |
| Testing | Pytest | 8.4.2 | Automated backend testing |
| Code Formatting | Black | 25.9.0 | Auto-code formatter |
| Linting | Flake8 | 7.3.0 | Syntax and style checker |

---

## 📂 Installation Path
All dependencies are installed inside:
 /Users/ajmalfahad/NEXO/backend/.venv/lib/python3.13/site-packages/

---

## ⚙️ Run Backend Server
cd /Users/ajmalfahad/NEXO/backend
source .venv/bin/activate
uvicorn main:app --reload

Access API docs at: http://127.0.0.1:8000/docs

---

## 🧩 Notes
- Environment fully isolated from global Python.
- macOS ARM (Apple Silicon) compatible.
- Ready for Expo/React Native integration.

