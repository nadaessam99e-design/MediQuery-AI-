import requests
import pandas as pd
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time

# ============================================================
# الإعدادات
# ============================================================
GEMINI_API_KEY = "AIza..."           # ← حط مفتاحك هنا
PERSIST_DIR    = "./drug_vectorstore"  # فولدر حفظ قاعدة البيانات

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# ============================================================
# الخطوة 1: جلب بيانات الأدوية
# ============================================================
def fetch_drug_data(limit=100):
    # بتروح على OpenFDA وبتجيب بيانات الأدوية
    print(f"⏳ بنجيب {limit} دواء...")
    url      = f"https://api.fda.gov/drug/label.json?limit={limit}"
    response = requests.get(url, timeout=30)
    data     = response.json()

    drugs = []
    for item in data.get("results", []):
        name = item.get("openfda", {}).get("brand_name", [""])
        name = name[0] if name else ""
        if not name:
            continue

        def get_field(key):
            val = item.get(key, [])
            return " ".join(val)[:500] if val else "غير متاح"

        drugs.append({
            "name"        : name,
            "purpose"     : get_field("purpose"),
            "warnings"    : get_field("warnings"),
            "dosage"      : get_field("dosage_and_administration"),
            "side_effects": get_field("adverse_reactions"),
            "interactions": get_field("drug_interactions"),
        })

    df = pd.DataFrame(drugs)
    print(f"✅ جبنا {len(df)} دواء!")
    return df


# ============================================================
# الخطوة 2: بناء قاعدة البيانات الذكية
# ============================================================
def build_vectorstore():
    # بيحول الكلام لأرقام عشان الكمبيوتر يقدر يدور فيه
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # لو قاعدة البيانات موجودة — حمّلها بدل ما نعملها من الأول
    if os.path.exists(PERSIST_DIR):
        print("✅ بنحمّل قاعدة البيانات الموجودة...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    # لو مش موجودة — نبنيها
    print("⏳ بنبني قاعدة البيانات لأول مرة...")
    df    = fetch_drug_data(100)
    texts = []

    for _, row in df.iterrows():
        texts.append(f"""اسم الدواء: {row['name']}
الاستخدام: {row['purpose']}
الجرعة: {row['dosage']}
الأعراض الجانبية: {row['side_effects']}
التحذيرات: {row['warnings']}
التفاعلات: {row['interactions']}""")

    # بنقطع النصوص لأجزاء صغيرة عشان البحث يبقى أدق
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs     = splitter.create_documents(texts)

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    print("✅ تم الحفظ!")
    return vectorstore


# بنشغل قاعدة البيانات لما الملف يتحمل
print("🚀 بنجهز الـ RAG...")
vectorstore = build_vectorstore()
retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ جاهز!")


# ============================================================
# الخطوة 3: الإجابة على السؤال
# ============================================================
def ask_drug_question(question: str) -> dict:
    start_time = time.time()

    # R — بيدور في قاعدة البيانات
    relevant_docs = retriever.invoke(question)
    context       = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    # A — بيضيف المعلومات للـ prompt
    prompt = f"""أنت صيدلاني خبير. أجب على السؤال بالعربي بشكل واضح.

المعلومات المتاحة:
{context}

السؤال: {question}

قواعد:
- استخدم المعلومات المتاحة فقط
- لو مش موجود قول بصراحة
- انصح بزيارة الطبيب للأمور المهمة"""

    # G — Gemini بيصيغ الإجابة
    response = model.generate_content(prompt)

    return {
        "answer"        : response.text,
        "response_time" : round(time.time() - start_time, 2),
        "sources_found" : len(relevant_docs)
    }
