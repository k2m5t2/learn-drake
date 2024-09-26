import sqlite3

class SQLiteReader:
    def __init__(self, db_path='shared_data.db'):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()

    def fetch_cube_positions(self, type="hands_only"):
        # Fetch the latest positions for all cubes from the database
        if type == "full_pose":
            self.c.execute("SELECT cube_id, x, y, z FROM cube_positions ORDER BY id DESC LIMIT 33")
        elif type == "hands_only":
            self.c.execute("SELECT cube_id, x, y, z FROM cube_positions ORDER BY id DESC LIMIT 2")
        elif type == "left_hand":
            self.c.execute("SELECT cube_id, x, y, z FROM cube_positions WHERE cube_id = 15 ORDER BY id DESC LIMIT 1")
        elif type == "right_hand":
            self.c.execute("SELECT cube_id, x, y, z FROM cube_positions WHERE cube_id = 16 ORDER BY id DESC LIMIT 1")
        # NOTE: dynamic SQL querying:
            # cube_ids = [1, 3]  # Example list of cube_ids
            # placeholders = ','.join('?' for _ in cube_ids)  # Create placeholders for SQL query
            # query = f"SELECT cube_id, x, y, z FROM cube_positions WHERE cube_id IN ({placeholders}) ORDER BY id DESC LIMIT 2"
        rows = self.c.fetchall()
        return rows

    def close(self):
        # Close the database connection
        self.conn.close()
