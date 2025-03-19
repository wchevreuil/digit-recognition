from datetime import datetime
import io
import psycopg2
import os
from PIL import Image

# PostgreSQL connection
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "digit_db"),
    "user": os.getenv("POSTGRES_USER", "user"),
    "password": os.getenv("POSTGRES_PASSWORD", "password"),
    "host": os.getenv("POSTGRES_HOST", "db"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def save_prediction(image_bytes, true_label, predicted_label, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()

    image = Image.open(io.BytesIO(image_bytes))
    # Convert image to binary
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_data = img_byte_arr.getvalue()

    # Insert into DB
    cursor.execute("""
        INSERT INTO pred_history (image, true_label, predicted_label, confidence, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (img_data, true_label, predicted_label, confidence, datetime.now()))

    conn.commit()
    cursor.close()
    conn.close()


def get_prediction_history(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""SELECT image, true_label, predicted_label, confidence, timestamp FROM pred_history ORDER BY 
                   timestamp DESC LIMIT %s""", (limit,))
    records = cursor.fetchall()

    cursor.close()
    conn.close()
    return records


def get_total_correct():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM pred_history WHERE true_label = predicted_label")
    count = int(cursor.fetchall()[0][0])
    cursor.close()
    conn.close()
    return count


def get_total():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM pred_history")
    count = int(cursor.fetchall()[0][0])
    cursor.close()
    conn.close()
    return count
