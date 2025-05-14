import pandas as pd
import sqlite3
from config import SHORT_TERM_DB, LONG_TERM_CSV

def perform_migration():
    conn = sqlite3.connect(SHORT_TERM_DB)
    df = pd.read_sql("SELECT * FROM short_term", conn)
    to_migrate = df[df['access_count'] < 2]
    if not to_migrate.empty:
        to_migrate.to_csv(LONG_TERM_CSV, mode='a', index=False, header=False)
        ids = tuple(to_migrate['id'])
        conn.execute(f"DELETE FROM short_term WHERE id IN {ids}")
        conn.commit()
    conn.close()
