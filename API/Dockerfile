FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y netcat-openbsd

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the model directory exists and copy the model file
RUN mkdir -p /app/model
COPY ../model/random_forest_model.pkl /app/model/random_forest_model.pkl

# Copy the wait-for-it script and make it executable
COPY wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

CMD ["./wait-for-it.sh", "db", "5432", "--", "python", "app.py"]