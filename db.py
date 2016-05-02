#!/usr/bin/env python3

import sys
import sqlite3
import numpy
from scipy import sparse
from itertools import count
from collections import defaultdict
from functools import singledispatch

class DBApi:
	def dump(self):
		print(self.__matrix)
		for index, user in enumerate(self.__user_map):
			print("User: " + str(index) + "->" + str(user))
		for index, item in enumerate(self.__item_map):
			print("Item: " + str(index) + "->" + str(item))

	def load(self, db_file, table):
		return DBApi.__load(db_file, self, table)

	@singledispatch
	def __load(db_file, self, table):
		raise NotImplementedError("Failed to recognize args")

	@__load.register(str)
	def _(db_file, self, table):
		print("db:", db_file, "table:", table)
		conn = sqlite3.connect(db_file)
		conn_cursor = conn.cursor()
		self.__to_matrix(conn_cursor, table)

	@__load.register(sparse.csc.csc_matrix)
	def _(matrix, self):
		print("sparse matrix")
	
	@__load.register(numpy.ndarray)
	def _(matrix, self):
		print("numpy array")

	def __to_matrix(self, conn_cursor, table):
		cols = [x[1] for x in conn_cursor.execute("PRAGMA table_info(" + table + ");").fetchall()[0:3]];
		users, items, quantities = zip(*conn_cursor.execute("SELECT " + ",".join(cols) + " FROM " +  table).fetchall())
		self.__user_map = list(set(users))
		self.__item_map = list(set(items))
		
		u_list = list(map(self.__user_map.index, users))
		i_list = list(map(self.__item_map.index, items))
		u_len  = len(self.__user_map)
		i_len  = len(self.__item_map)
		
		self.__matrix = sparse.csc_matrix((quantities, (u_list, i_list)), shape=(u_len, i_len))

if __name__ == "__main__":
	try:
		db_file = sys.argv[1]
		table = sys.argv[2]
	except IndexError:
		print("Usage: ./db.py [db] [table]")
		sys.exit(1)
	db_api = DBApi()
	db_api.load(db_file, table)
	db_api.dump()