import sqlite3
from datetime import datetime
import json
# Function to get conversation history
def get_conversation_history(thread_id):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE thread_id = ? ORDER BY timestamp", (thread_id,))
        return cursor.fetchall()

# Initialize the database tables
def initialize_db():
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                timestamp TEXT,
                role TEXT,
                content TEXT
            )
        """)
        conn.commit()

# Function to store conversation in memory
def store_conversation(thread_id, role, content):
    with sqlite3.connect("agent_memory.db") as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT INTO conversations (thread_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
        """, (thread_id, timestamp, role, content))
        conn.commit()

# Main execution block
if __name__ == "__main__":
    print("Initializing database...")
    initialize_db()
    print("Database initialized successfully.")