FROM python:3.11-slim
 
WORKDIR /app
 
# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy source
COPY . .
 
# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser
 
EXPOSE 8000
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
 