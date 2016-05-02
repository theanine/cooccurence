#!/usr/bin/env python3

import sys
import sqlite3
import numpy
from scipy import sparse
from functools import singledispatch

class DBApi:

	@classmethod
	def to_matrix(self, users, items, quantities):
		user_map = list(set(users))
		item_map = list(set(items))
		
		u_list = list(map(user_map.index, users))
		i_list = list(map(item_map.index, items))
		u_len  = len(user_map)
		i_len  = len(item_map)
		
		matrix = sparse.csc_matrix((quantities, (u_list, i_list)), shape=(u_len, i_len))
		return user_map, item_map, matrix

	def dump(self):
		print(self.__matrix)
		if self.__user_map:
			for index, user in enumerate(self.__user_map):
				print("User: " + str(index) + "->" + str(user))
		if self.__item_map:
			for index, item in enumerate(self.__item_map):
				print("Item: " + str(index) + "->" + str(item))

	def load(self, db_file, table=False):
		if table:
			return DBApi.__load(db_file, self, table)
		return DBApi.__load(db_file, self)

	@singledispatch
	def __load(db_file, self, table=False):
		raise NotImplementedError("Failed to recognize args")

	@__load.register(str)
	def _(db_file, self, table=False):
		print("db:", db_file, "table:", table)
		conn = sqlite3.connect(db_file)
		conn_cursor = conn.cursor()
		self.__from_sql(conn_cursor, table)

	@__load.register(sparse.csc_matrix)
	def _(matrix, self):
		self.__matrix = matrix
		self.__user_map = None
		self.__item_map = None

	@__load.register(numpy.ndarray)
	def _(matrix, self):
		self.__matrix = sparse.csc_matrix(numpy.asmatrix(matrix))
		self.__user_map = None
		self.__item_map = None

	def __from_sql(self, conn_cursor, table):
		cols = [x[1] for x in conn_cursor.execute("PRAGMA table_info(" + table + ");").fetchall()[0:3]];
		users, items, quantities = zip(*conn_cursor.execute("SELECT " + ",".join(cols) + " FROM " +  table).fetchall())
		self.__user_map, self.__item_map, self.__matrix = self.to_matrix(users, items, quantities)

def test_cli():
	try:
		db_file = sys.argv[1]
		table = sys.argv[2]
	except IndexError:
		print("Usage: ./db.py [db] [table]")
		sys.exit(1)
	db_api = DBApi()
	db_api.load(db_file, table)
	db_api.dump()

def test_sparse():
	row = numpy.array([9999, 9999, 1, 2, 2, 2])
	col = numpy.array([9999, 2, 2, 9999, 1, 2])
	data = numpy.array([1, 2, 3, 4, 5, 6])
	matrix = sparse.csc_matrix((data, (row, col)), shape=(10000, 10000))
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

def test_numpy():
	matrix = numpy.array(([[1, 0, 2],
                           [0, 0, 3],
                           [4, 5, 6]]))
	print(matrix)
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

if __name__ == "__main__":
	test_cli()
	test_sparse()
	test_numpy()
