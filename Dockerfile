FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default: run the interactive demo
CMD ["python", "-m", "scripts.demo"]
