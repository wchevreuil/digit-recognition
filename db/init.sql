CREATE TABLE IF NOT EXISTS pred_history (
    id SERIAL PRIMARY KEY,
    image BYTEA,   -- Store image as binary
    true_label INT,
    predicted_label INT,
    confidence FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);