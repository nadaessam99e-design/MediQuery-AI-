from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import ask_drug_question
import mlflow

app = FastAPI(
    title       = "Drug Chatbot API",
    description = "مساعد الأدوية الذكي بالـ RAG + Gemini",
    version     = "1.0.0"
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("drug-chatbot")

class Question(BaseModel):
    question: str


@app.get("/")
def root():
    # بيتأكد إن الـ API شغال
    return {"status": "running", "message": "Drug Chatbot API شغال ✅"}


@app.get("/health")
def health():
    # AWS بيستخدمه عشان يتأكد إن الـ container تمام
    return {"status": "healthy"}


@app.post("/ask")
def ask(q: Question):
    # بياخد السؤال ويرجع الإجابة
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="السؤال فاضي!")

    result = ask_drug_question(q.question)

    # بيسجل في MLflow عشان نتتبع الأداء
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_metric("response_time", result["response_time"])
            mlflow.log_metric("sources_found", result["sources_found"])
            mlflow.log_param("question_length", len(q.question))
    except Exception:
        pass  # لو MLflow مش شغال، منوقفش الـ API

    return {
        "question"             : q.question,
        "answer"               : result["answer"],
        "response_time_seconds": result["response_time"],
        "sources_found"        : result["sources_found"]
    }
