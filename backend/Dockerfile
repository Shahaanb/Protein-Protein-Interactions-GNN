FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The CMD assumes we're running via docker-compose which might bind-mount things,
# but we provide a default command.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
