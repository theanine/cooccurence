#!/usr/bin/env python

import sys
import sqlite3
from scipy import sparse

def to_matrix(cur, table):
	numrows = cur.execute("SELECT COUNT(DISTINCT userid) FROM " + table).fetchone()[0]
	numcols = cur.execute("SELECT COUNT(DISTINCT itemid) FROM " + table).fetchone()[0]

	users, items, quantities = zip(*cur.execute("SELECT * FROM " +  table).fetchall())
	mat = sparse.csr_matrix((quantities, (users, items)), shape=(numrows, numcols))
	print mat

def main():
	try:
		db_name = sys.argv[1]
		table = sys.argv[2]
	except IndexError:
		print "No DB or table provided!\n./db.py [db] [table]"
		sys.exit(1)

	print "db:", db_name, "table:", table

	conn = sqlite3.connect(db_name)
	cur = conn.cursor()
	to_matrix(cur, table)

if __name__ == "__main__":
	main()