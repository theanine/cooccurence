#!/usr/bin/env python

import sys
import numpy as np
from scipy import spatial
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
PREDICTION_MODEL = "a"

def matrix_to_perc(matrix):
	diagonal = np.diagonal(matrix)
	with np.errstate(divide='ignore', invalid='ignore'):
		matrix = np.nan_to_num(np.true_divide(matrix, diagonal[:, None]))
	return matrix

def matrix_zero_diag(matrix):
	np.fill_diagonal(matrix, 0)
	return matrix

def matrix_cos_similarity(matrix):
	# squared magnitude of preference vectors
	square_mag = np.diagonal(matrix)
	
	# inverse squared magnitude
	with np.errstate(divide='ignore', invalid='ignore'):
		inv_square_mag = 1 / square_mag

	# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
	inv_square_mag[np.isinf(inv_square_mag)] = 0

	# inverse of the magnitude
	inv_mag = np.sqrt(inv_square_mag)

	# cosine similarity (elementwise multiply by inverse magnitudes)
	cosine = matrix * inv_mag
	cosine = cosine.T * inv_mag
	return cosine

def cooccurence(arr, cos_similarity, in_perc, zero_diag):
	items = np.array(arr).astype(float)
	# TODO: replace this with items.dot(items.T).todense() for sparse representation?
	#       items also must be initialized as a scipy.sparse type
	matrix = np.dot(items.T, items)
	if cos_similarity:
		matrix = matrix_cos_similarity(matrix)
	if in_perc:
		matrix = matrix_to_perc(matrix)
	if zero_diag:
		matrix = matrix_zero_diag(matrix)
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
	
	(zero_diag, in_perc, cos_similarity) = switcher.get(PREDICTION_MODEL, None)
	
	cooccur = cooccurence(data, cos_similarity, in_perc, zero_diag)
	
	prediction = np.dot(cooccur, target)
	
	if not cos_similarity:
		prediction = np.true_divide(prediction, len(data))
	
	return np.round(prediction, 10)

def test1():
	data = [[1, 0, 1, 0],
	        [0, 1, 1, 0],
	        [0, 1, 0, 0]]
	print("data:")
	print(np.array(data))

	for i in range(0,3):
		target = [0, 0, 0, 0]
		target[i] = 1

		prediction = predict(data, target)

		print("target:")
		print(np.array(target))

		print("prediction:")
		print(prediction)

def test2():
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

def test3():
	data = [[1, 0, 1, 0],
	        [0, 1, 1, 0],
	        [0, 1, 0, 0]]
	print("data:")
	print(np.array(data))

	cooccur = cooccurence(data, False, False, False)

	print("cooccur:")
	print(cooccur)

if __name__ == "__main__":
	test1()
	test2()
	test3()