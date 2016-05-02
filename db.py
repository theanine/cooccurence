#!/usr/bin/env python3

import sys
import sqlite3
import numpy
from scipy import sparse
from functools import singledispatch

class DBApi:
	def dump(self):
		print(self.__matrix)
		for index, user in enumerate(self.__user_map):
			print("User: " + str(index) + "->" + str(user))
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
		self.__to_matrix(conn_cursor, table)

	@__load.register(sparse.csc_matrix)
	def _(matrix, self):
		print("sparse matrix")
		self.__matrix = matrix
		self.__user_map = range(matrix.shape[0])
		self.__item_map = range(matrix.shape[1])

	@__load.register(numpy.ndarray)
	def _(matrix, self):
		self.__matrix = sparse.csc_matrix(numpy.asmatrix(matrix))
		self.__user_map = range(matrix.shape[0])
		self.__item_map = range(matrix.shape[1])

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
	row = numpy.array([0, 0, 1, 2, 2, 2])
	col = numpy.array([0, 2, 2, 0, 1, 2])
	data = numpy.array([1, 2, 3, 4, 5, 6])
	matrix = sparse.csc_matrix((data, (row, col)), shape=(3, 3))
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

def test_numpy():
	row = numpy.array([0, 0, 1, 2, 2, 2])
	col = numpy.array([0, 2, 2, 0, 1, 2])
	data = numpy.array([1, 2, 3, 4, 5, 6])
	matrix = sparse.csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

if __name__ == "__main__":
	test_cli()
	test_sparse()
	test_numpy()
