#!/usr/bin/env python

import sys
import sqlite3
from scipy import sparse
from itertools import count
from collections import defaultdict

class DBApi:
	def dump(self):
		print self.__matrix
		for index, user in enumerate(self.__user_map):
			print("User: " + str(index) + "->" + str(user))
		for index, item in enumerate(self.__item_map):
			print("Item: " + str(index) + "->" + str(item))

	def load(self, db_file, table):
		print "db:", db_file, "table:", table
		conn = sqlite3.connect(db_file)
		cursor = conn.cursor()
		self.__to_matrix(cursor, table)

	def __to_matrix(self, cursor, table):
		cols = [x[1] for x in cursor.execute("PRAGMA table_info(" + table + ");").fetchall()[0:3]];
		users, items, quantities = zip(*cursor.execute("SELECT " + ",".join(cols) + " FROM " +  table).fetchall())
		self.__user_map = list(set(users))
		self.__item_map = list(set(items))
		self.__matrix = sparse.csc_matrix(
			(quantities, (map(self.__user_map.index, users), map(self.__item_map.index, items))),
			shape=(len(self.__user_map), len(set(self.__item_map))))

if __name__ == "__main__":
	try:
		db_file = sys.argv[1]
		table = sys.argv[2]
	except IndexError:
		print "Usage: ./db.py [db] [table]"
		sys.exit(1)
	db_api = DBApi()
	db_api.load(db_file, table)
	db_api.dump()