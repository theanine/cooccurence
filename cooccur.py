#!/usr/bin/env python3

import sys
import numpy as np
from scipy import spatial, sparse
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
PREDICTION_MODEL = "c"

def to_pseudo_unique(data):
	arr = np.array(data)
	maxes = []
	num_cols = len(arr[0])
	for i in range(num_cols):
		m = max(arr[:,i])
		maxes.append(m if m != 0 else 1)
	new_arr = np.zeros((len(arr), sum(maxes)))
	rows, cols = arr.nonzero()
	num_nonzero = len(rows)
	for i in range(num_nonzero):
		val = arr[rows[i], cols[i]]
		for offset in range(val):
			new_col = (sum(maxes[:cols[i]]) + offset)
			new_arr[rows[i], new_col] = 1
	return new_arr

def matrix_to_perc(matrix):
	diagonal = matrix.diagonal()
	rows, cols = matrix.nonzero()
	num_points = len(rows)
	for i in range(num_points):
		d = diagonal[rows[i]]
		if d != 0:
			matrix[rows[i], cols[i]] /= d
		else:
			matrix[rows[i], cols[i]] = 0
	return matrix

def matrix_zero_diag(matrix):
	matrix.setdiag(0)
	return matrix

def matrix_cos_similarity(matrix):
	# squared magnitude of preference vectors
	square_mag = matrix.diagonal()
	
	# inverse squared magnitude
	with np.errstate(divide='ignore', invalid='ignore'):
		inv_square_mag = 1 / square_mag

	# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
	inv_square_mag[np.isinf(inv_square_mag)] = 0

	# inverse of the magnitude
	inv_mag = np.sqrt(inv_square_mag)

	#convert to sparse diag matrix
	inv_mag = sparse.spdiags(inv_mag, 0, len(inv_mag), len(inv_mag))

	# cosine similarity (elementwise multiply by inverse magnitudes)
	cosine = matrix * inv_mag
	cosine = cosine.T * inv_mag
	return cosine

def cooccurence(arr, cos_similarity, in_perc, zero_diag):
	items = sparse.csc_matrix(arr).astype(float)

	matrix = items.T.dot(items)

	if cos_similarity:
		matrix = matrix_cos_similarity(matrix)
	if in_perc:
		matrix = matrix_to_perc(matrix)
	if zero_diag:
		matrix = matrix_zero_diag(matrix)

	matrix.eliminate_zeros()
	return matrix

def predict(data, target):	
	switcher = {
		"a": (True,  True,  True),
		"b": (True,  True,  False),
		"c": (True,  False, True),
		"d": (True,  False, False),
		"e": (False, True,  True),
		"f": (False, True,  False),
		"g": (False, False, True),
		"h": (False, False, False),
	}
	
	data = to_pseudo_unique(data)

	(zero_diag, in_perc, cos_similarity) = switcher.get(PREDICTION_MODEL, None)
	
	cooccur = cooccurence(data, cos_similarity, in_perc, zero_diag)
	
	prediction = cooccur.dot(target)
	#prediction is now an nd-matrix
	
	if not cos_similarity:
		prediction = np.true_divide(prediction, len(data))
	
	return np.round(prediction, 2)

def test1():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	data = np.array(
		[[1, 1, 0, 1],
		 [2, 0, 1, 1]])
	
	print("data:")
	print(data)
	
	dist_out = 1 - pairwise_distances(data, metric="cosine")
	print("cos_sim:")
	print(dist_out)

	items = np.array(data).astype(float)
	matrix = np.dot(items.T, items)
	matrix = np.dot(items, items.T)
	print("items:")
	print(items)
	print("items.T:")
	print(items.T)
	print("matrix:")
	print(matrix)
	
	# matrix = np.array(data).astype(float)
	# cos_sim = matrix_cos_similarity(matrix)
	
	# print("cos_sim:")
	# print(cos_sim)

def test2():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	data = [[1, 0, 1, 0],
	        [0, 1, 1, 0],
	        [0, 1, 0, 0]]
	print("data:")
	print(np.array(data))

	cooccur = cooccurence(data, False, False, False)

	print("cooccur:")
	print(cooccur)

def test3():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	data = [[1, 0, 1, 0],
	        [0, 1, 1, 0],
	        [0, 1, 0, 0]]
	print("data:")
	print(np.array(data))

	for i in range(0,4):
		target = [0, 0, 0, 0]
		target[i] = 1

		prediction = predict(data, target)
		print("target:     " + str(np.round(np.array(target), 1)))
		print("prediction: " + str(prediction))

def test4():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	data = [[1, 0, 2, 0],
	        [0, 1, 1, 0],
	        [0, 1, 0, 0]]
	print("data:")
	print(np.array(data))

	for i in range(0,4):
		target = [0, 0, 0, 0]
		target[i] = 1

		prediction = predict(data, target)
		print("target:     " + str(np.round(np.array(target), 1)))
		print("prediction: " + str(prediction))

def test5():
	print("==== " + sys._getframe().f_code.co_name + " ====")
	data = [[1, 0, 1, 1, 0],
	        [0, 1, 1, 0, 0],
	        [0, 1, 0, 0, 0]]
	print("data:")
	print(np.array(data))

	for i in range(0,5):
		target = [0, 0, 0, 0, 0]
		target[i] = 1

		prediction = predict(data, target)
		print("target:     " + str(np.round(np.array(target), 1)))
		print("prediction: " + str(prediction))

if __name__ == "__main__":
	for model in ("a" "b" "c" "d" "e" "f" "g" "h"):
		PREDICTION_MODEL = model
		print("Prediction model: " + model)
		# test1()
		# test2()
		test3()
		# test4()
		# test5()
		