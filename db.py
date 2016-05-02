#!/usr/bin/env python

import sys
import sqlite3
from scipy import sparse
from itertools import count
from collections import defaultdict

def dump(matrix, user_map, item_map):
	print matrix
	for index, user in enumerate(user_map):
		print("User: " + str(index) + "->" + str(user))
	for index, item in enumerate(item_map):
		print("Item: " + str(index) + "->" + str(item))

def to_matrix(cur, table):
	cols = [x[1] for x in cur.execute("PRAGMA table_info(" + table + ");").fetchall()[0:3]];

	users, items, quantities = zip(*cur.execute("SELECT " + ",".join(cols) + " FROM " +  table).fetchall())
	user_map = list(set(users))
	item_map = list(set(items))
	mat = sparse.csc_matrix((quantities, (map(user_map.index, users), map(item_map.index, items))), shape=(len(user_map), len(set(item_map))))
	dump(mat, user_map, item_map)

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