FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Build data + model artifacts at image build time so the image is self-contained
RUN cd src && python generate_data.py && python etl.py && python models.py

EXPOSE 8000 8501
# API:       docker run -p 8000:8000 IMAGE
# Streamlit: docker run -p 8501:8501 IMAGE streamlit run app.py --server.address 0.0.0.0
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
