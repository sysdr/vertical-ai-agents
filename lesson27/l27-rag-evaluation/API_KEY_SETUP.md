# API Key Setup for L27 RAG Evaluation

The evaluation demo requires a valid **Gemini API key** to run RAG evaluations and get **real RAGAS metrics** (Faithfulness, Relevance, Precision, Recall computed by AI).

## Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Edit `backend/.env` and set:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. Restart the backend if it's already running

## Without API Key

- **Dashboard**: Works - shows demo metrics when you run evaluation
- **Run Evaluation**: Uses demo fallback (varied synthetic scores per run)
- **Metrics**: Update with each run but show synthetic values; set API key for real RAGAS scores

## With API Key

- **Run Evaluation**: Click "Run Evaluation" to execute RAGAS metrics on sample queries
- **Metrics**: Dashboard updates with real scores from completed runs
- **Trends**: Chart shows evaluation run results
