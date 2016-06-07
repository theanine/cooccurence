#!/usr/bin/env python3

import sys
import sqlite3
import numpy as np
from scipy import sparse
from functools import singledispatch
import cooccur

class DBApi:

	@classmethod
	def to_matrix(self, users, items, quantities):
		user_map = dict(enumerate(set(users)))
		item_map = dict(enumerate(set(items)))
		
		user_to_index_map = dict(map(reversed, user_map.items()))
		item_to_index_map = dict(map(reversed, item_map.items()))

		u_list = list(map(user_to_index_map.get, users))
		i_list = list(map(item_to_index_map.get, items))
		u_len  = len(user_map)
		i_len  = len(item_map)
		
		matrix = sparse.csc_matrix((quantities, (u_list, i_list)), shape=(u_len, i_len))
		return user_map, item_map, matrix

	def convert_to_pseudo_unique(self):
		arr = self.__matrix

		# get max value in each column
		maxes = []
		num_cols = arr.shape[1]
		for i in range(num_cols):
			m = arr[:,i].max()
			maxes.append(m if m != 0 else 1)

		new_arr = sparse.csc_matrix((arr.shape[0], sum(maxes)))
		new_item_map = {}

		# generate new matrix
		rows, cols = arr.nonzero()
		num_nonzero = len(rows)
		for i in range(num_nonzero):
			val = arr[rows[i], cols[i]]
			new_base_col = sum(maxes[:cols[i]])
			for offset in range(val):
				new_offset_col = (new_base_col + offset)
				new_arr[rows[i], new_offset_col] = 1

		# make new item map
		new_col = 0
		for col, m in enumerate(maxes):
			new_item_map[new_col] = self.__index_to_item_map[col] if self.__index_to_item_map else col
			new_col += m

		self.update(self.__user_map, new_item_map, new_arr)

	def update(self, user_map, index_to_item_map, matrix):
		self.__matrix = matrix
		self.__user_map = user_map
		self.__index_to_item_map = index_to_item_map
		self.__item_to_index_map = dict(map(reversed, index_to_item_map.items())) if index_to_item_map else None
		self.__item_to_len_map = None

		if self.__item_to_index_map:
			item_cols = sorted(self.__item_to_index_map.values())
			pairs = zip(item_cols, item_cols[1:]+[self.__matrix.shape[1]])
			maxes = list(map(lambda pair:pair[1] - pair[0], pairs))
			self.__item_to_len_map = {}
			for i in range(len(maxes)):
				item = self.__index_to_item_map[sum(maxes[:i])]
				self.__item_to_len_map[item] = maxes[i]

	def dump(self):
		print(self.__matrix.todense())
		if self.__user_map:
			for index, user in self.__user_map.items():
				print("Index: " + str(index) + "-> User:" + str(user))
		if self.__index_to_item_map:
			for index, item in self.__index_to_item_map.items():
				print("Index: " + str(index) + "-> Item:" + str(item))

	def load(self, db_file, table=False):
		if table:
			DBApi.__load(db_file, self, table)
		else:
			DBApi.__load(db_file, self)
		self.convert_to_pseudo_unique()

	def predict(self, target):
		target_mat = self.target_map_to_matrix(target)
		return cooccur.predict(self.__matrix, target_mat)

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
		self.update(None, None, matrix)

	@__load.register(np.ndarray)
	def _(matrix, self):
		self.update(None, None, sparse.csc_matrix(np.asmatrix(matrix)))

	def __from_sql(self, conn_cursor, table):
		cols = [x[1] for x in conn_cursor.execute("PRAGMA table_info(" + table + ");").fetchall()[0:3]];
		users, items, quantities = zip(*conn_cursor.execute("SELECT " + ",".join(cols) + " FROM " +  table).fetchall())
		self.update(*self.to_matrix(users, items, quantities))

	def target_map_to_matrix(self, target):
		mat = np.zeros((self.__matrix.shape[1], 1))
		for item, val in target.items():
			i = self.__item_to_index_map[item]

			# trim data
			v = val
			if val > self.__item_to_len_map[item]:
				v = self.__item_to_len_map[item]

			# generate new matrix
			for offset in range(v):
				mat[(i + offset), 0] = 1

		return mat

def test_db():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	db_file = "test.db"
	table = "useritems"
	db_api = DBApi()
	db_api.load(db_file, table)

	test = {0:1}
	prediction = db_api.predict(test)
	assert(prediction[0] == 0)
	assert(prediction[1] == 0)
	assert(prediction[2] == 0.71)
	for i in range(3, 6667):
		assert(prediction[i] == 0)

	test = {1:1}
	prediction = db_api.predict(test)
	assert(prediction[0] == 0)
	assert(prediction[1] == 0)
	assert(prediction[2] == 0.5)
	for i in range(3, 6667):
		assert(prediction[i] == 0)

	test = {2:1}
	prediction = db_api.predict(test)
	assert(prediction[0] == 0.71)
	assert(prediction[1] == 0.5)
	assert(prediction[2] == 0)
	for i in range(3, 6667):
		assert(prediction[i] == 0)

	test = {98765:1}
	prediction = db_api.predict(test)
	assert(prediction[0] == 0)
	assert(prediction[1] == 0)
	assert(prediction[2] == 0)
	assert(prediction[3] == 0)
	for i in range(4, 6667):
		assert(prediction[i] == 1)

# TODO: update this test to use asserts
def test_sparse():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	row = np.array( [9999, 9999, 1,    2, 2, 2])
	col = np.array( [9999,    2, 2, 9999, 1, 2])
	data = np.array([   1,    2, 3,    4, 5, 6])
	matrix = sparse.csc_matrix((data, (row, col)), shape=(10000, 10000))
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

# TODO: update this test to use asserts
def test_np():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	matrix = np.array(([[2, 0, 2, 0],
	                    [0, 1, 1, 0],
	                    [0, 1, 0, 3]]))
	print(matrix)
	print(type(matrix))
	db_api = DBApi()
	db_api.load(matrix)
	db_api.dump()

if __name__ == "__main__":
	# test_sparse()
	# test_np()
	test_db()
