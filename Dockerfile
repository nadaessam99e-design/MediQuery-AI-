# بنبدأ من Python 3.11 نظيفة وصغيرة
FROM python:3.11-slim

# الفولدر اللي هيشتغل فيه الكود جوه الـ container
WORKDIR /app

# بننسخ requirements الأول — عشان Docker يـ cache الـ packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# بننسخ باقي الكود
COPY . .

# الـ container بيسمع على البورت 8000
EXPOSE 8000

# الأمر اللي بيشتغل لما الـ container يقوم
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
