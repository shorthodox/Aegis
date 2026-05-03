import sqlite3
from datetime import datetime, timedelta
import os

class DBHandler:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "subscribers.db")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            tier TEXT DEFAULT 'basic',
            trial_end DATETIME,
            subscription_end DATETIME,
            mode TEXT DEFAULT 'safe',
            balance REAL DEFAULT 1000
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def get_user(self, user_id):
        cursor = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return cursor.fetchone()

    def add_user(self, user_id):
        trial_end = datetime.now() + timedelta(days=3)
        self.conn.execute("INSERT OR IGNORE INTO users (user_id, trial_end) VALUES (?, ?)", 
                          (user_id, trial_end))
        self.conn.commit()

    def update_mode(self, user_id, mode):
        self.conn.execute("UPDATE users SET mode = ? WHERE user_id = ?", (mode, user_id))
        self.conn.commit()

    def update_balance(self, user_id, balance):
        self.conn.execute("UPDATE users SET balance = ? WHERE user_id = ?", (balance, user_id))
        self.conn.commit()

    def is_subscription_active(self, user_id):
        user = self.get_user(user_id)
        if not user: return False
    
        # user_data = (id, tier, trial_end, sub_end, mode, balance)
        trial_end = datetime.strptime(user[2], '%Y-%m-%d %H:%M:%S') if user[2] else None
        sub_end = datetime.strptime(user[3], '%Y-%m-%d %H:%M:%S') if user[3] else None
        now = datetime.now()

        # If they have an active sub, or they are still in their 3-day trial
        if (sub_end and sub_end > now) or (trial_end and trial_end > now):
         return True
         return False