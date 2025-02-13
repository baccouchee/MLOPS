FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the data directory has the correct permissions
RUN mkdir -p /app/data && chmod -R 777 /app/data

# Execute the cleaning script
CMD ["python", "scripts/cleaning_data.py"]