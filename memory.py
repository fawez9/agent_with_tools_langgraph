import sqlite3
from datetime import datetime

class DatabaseHandler:
    def __init__(self, db_name="agent_memory.db"):
        self.db_name = db_name
        self.initialize_db()

    def initialize_db(self):
        with sqlite3.connect(self.db_name) as conn:
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

    def store_conversation(self, thread_id, role, content):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute("""
            INSERT INTO conversations (thread_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
            """, (thread_id, timestamp, role, content))
            conn.commit()

    def get_conversation_history(self, thread_id):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE thread_id = ? ORDER BY timestamp", (thread_id,))
            return cursor.fetchall()

    def load_sessions(self):
        sessions = {}
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM conversations")
            for row in cursor.fetchall():
                sessions[row[0]] = "Active"
        return sessions

