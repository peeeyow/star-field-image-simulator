from pathlib import Path
from os.path import exists
import sqlite3

from .constants import DATA_PATH


conn = sqlite3.connect(DATA_PATH / "sc_no_loop.db")
curs = conn.cursor()
curs.execute("SELECT * FROM star_catalog")
print(curs.fetchall())
