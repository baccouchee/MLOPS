from flask import Flask, request, jsonify
import pickle
import pandas as pd
import psycopg2
import json

app = Flask(__name__)

# Load the trained model
model_path = 'model/random_forest_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Database connection
conn = psycopg2.connect(
    dbname="predictions",
    user="user",
    password="password",
    host="db"
)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    
    # Store prediction in the database
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO predictions (input_data, prediction)
    VALUES (%s, %s)
    """, (json.dumps(data), prediction.tolist()))
    conn.commit()
    cur.close()
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)