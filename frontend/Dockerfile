FROM python:3.12-slim
WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY api/utils/logger.py ./api/utils/logger.py
COPY frontend/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
