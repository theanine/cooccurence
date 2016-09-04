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
		# TODO: this code is shit, rewrite it.
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
			index = cols[i]
			new_base_col = sum(maxes[:index])
			for offset in range(val):
				new_offset_col = (new_base_col + offset)
				# TODO: Changing the sparsity structure of a csc_matrix is expensive. lil_matrix is more efficient.
				new_arr[rows[i], new_offset_col] = 1
				item = self.__index_to_item_map[index]
				new_item_map[new_offset_col] = item

		self.update(self.__user_map, new_item_map, new_arr)

	def update(self, user_map, index_to_item_map, matrix):
		self.__matrix = matrix
		self.__user_map = user_map
		self.__index_to_item_map = index_to_item_map
		self.__item_to_index_map = {}
		for k, v in index_to_item_map.items():
			if v not in self.__item_to_index_map.keys():
				self.__item_to_index_map[v] = k
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
		if self.__item_to_index_map:
			for index, item in self.__item_to_index_map.items():
				print("Item: " + str(index) + "-> Index:" + str(item))

	def load(self, db_file, table=False):
		if table:
			DBApi.__load(db_file, self, table)
		else:
			DBApi.__load(db_file, self)
		self.convert_to_pseudo_unique()

	def predict(self, target):
		target_mat = self.target_map_to_matrix(target)
		prediction = cooccur.predict(self.__matrix, target_mat)
		prediction = {i: prediction[i][0] for i in range(len(prediction))}
		prediction = {x:y for x,y in prediction.items() if y!=0}
		if self.__index_to_item_map:
			if len(prediction.items()) < 10:
				for x,y in prediction.items():
					print(x, "=>", y)
			prediction = {self.__index_to_item_map[x]:y for x,y in prediction.items()}
		# NOTE: should we sort the list somehow? 
		# prediction = sorted(prediction, key=prediction.__getitem__, reverse=True)
		return prediction

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
		# zip(*a) is matrix transposition
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

def test_db_nums():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	db_file = "test.db"
	table = "useritems"
	db_api = DBApi()
	db_api.load(db_file, table)

	test = {0:1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 2)
		assert(v == 0.71)

	test = {1:1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 2)
		assert(v == 0.5)

	test = {2:1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 2)
	for k in prediction:
		v = prediction[k]
		assert(k == 0 or k == 1)
		if k == 0:
			assert(v == 0.71)
		if k == 1:
			assert(v == 0.5)

	test = {98765:1}
	prediction = db_api.predict(test)
	db_api.dump()
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 98765)
		assert(v == 1)

	print("PASSED")

def test_db_words():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	db_file = "test.db"
	table = "worditems"
	db_api = DBApi()
	db_api.load(db_file, table)

	test = {'a':1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 'foo')
		assert(v == 0.71)

	test = {'trist':1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 'foo')
		assert(v == 0.5)

	test = {'foo':1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 2)
	for k in prediction:
		v = prediction[k]
		assert(k == 'a' or k == 'trist')
		if k == 'a':
			assert(v == 0.71)
		if k == 'trist':
			assert(v == 0.5)

	test = {'peesy':1}
	prediction = db_api.predict(test)
	assert(len(prediction) == 1)
	for k in prediction:
		v = prediction[k]
		assert(k == 'peesy')
		assert(v == 1)

	print("PASSED")

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

def run_tests():
	# test_sparse()
	# test_np()
	test_db_nums()
	test_db_words()

if __name__ == "__main__":
	run_tests()
	
	# db_file = "test.db"
	# table = "useritems"
	# db_api = DBApi()
	# db_api.load(db_file, table)

	# test = {0:1}
	# prediction = db_api.predict(test)
	# print(prediction)
