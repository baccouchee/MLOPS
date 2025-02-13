import psycopg2

conn = psycopg2.connect(
    dbname="predictions",
    user="user",
    password="password",
    host="db"
)

cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    input_data JSONB,
    prediction VARCHAR(50)
)
""")
conn.commit()
cur.close()
conn.close()