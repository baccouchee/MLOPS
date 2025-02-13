FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# Execute the cleaning script
CMD ["python", "scripts/cleaning_data.py"]