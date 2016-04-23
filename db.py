#!/usr/bin/env python

import sys
import sqlite3
import numpy as np
from scipy import sparse

def to_matrix(cur, table):
	numrows = cur.execute("SELECT COUNT(DISTINCT userid) FROM " + table).fetchone()
	numcols = cur.execute("SELECT COUNT(DISTINCT itemid) FROM " + table).fetchone()

	users = cur.execute("SELECT userid FROM " +  table).fetchall()
	items = cur.execute("SELECT itemid FROM " +  table).fetchall()
	quantities = cur.execute("SELECT quantity FROM " +  table).fetchall()
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
	conn.row_factory = lambda cursor, row: row[0]
	cur = conn.cursor()
	to_matrix(cur, table)

if __name__ == "__main__":
	main()