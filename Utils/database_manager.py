from langgraph.graph import StateGraph, START, END
# Imports for SQLite checkpointer
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Setup local SQLite files (they will auto-generate in your project folder)
chk_conn = sqlite3.connect("banking_checkpoints.db", check_same_thread=False)
my_checkpointer = SqliteSaver(chk_conn)

# For the Store component, if using a custom SQLite wrapper or InMemoryStore backed by a file:
# LangGraph's native InMemoryStore can also be initialized, 
# or you can hook up your own lightweight SQLite table for memories:

import sqlite3
import json

class SqliteKeyValueStore:
    """A lightweight persistent LTM store using a local SQLite file."""
    def __init__(self, db_path="ltm_store.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    namespace TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (namespace, key)
                )
            """)

    def put(self, namespace: tuple, key: str, value: dict):
        ns_str = ".".join(namespace)
        val_str = json.dumps(value)
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO memories (namespace, key, value)
                VALUES (?, ?, ?)
            """, (ns_str, key, val_str))

    def search(self, namespace: tuple):
        ns_str = ".".join(namespace)
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM memories WHERE namespace = ?", (ns_str,))
        rows = cursor.fetchall()
        
        # Format to match LangGraph Store item structure (mocking an object with a .value attribute)
        class Item:
            def __init__(self, val):
                self.value = val
                
        return [Item(json.loads(row[1])) for row in rows]

# Initialize our permanent file-based store
ltm_store = SqliteKeyValueStore("banking_ltm.db")