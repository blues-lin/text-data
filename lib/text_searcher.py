import sqlite3

class TextSearcher:
    """ Search documents related to input term. """


    def __init__(self, sqlitePath):
        conn = sqlite3.connect(sqlitePath)
        self._cur = conn.cursor()

        # For getting sqlite table:
        # cur.execute("SELECT * from SQLITE_MASTER")
        # c = cur.fetchall()


    def genDocs(self, term):
        "Generate text with term included."
        articles = self._cur.execute("SELECT article from corpus")
        for art in articles:
            if term in art[0]:
                yield art[0]
