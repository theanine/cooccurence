#!/usr/bin/env python

import sys
import numpy as np
from scipy import spatial

PREDICTION_MODEL = "a"

def matrix_to_perc(matrix):
	diagonal = np.diagonal(matrix)
	print(diagonal)
	with np.errstate(divide='ignore', invalid='ignore'):
		matrix = np.nan_to_num(np.true_divide(matrix, diagonal[:, None]))
	return matrix

def matrix_zero_diag(matrix):
	return np.fill_diagonal(matrix, 0)

def matrix_cos_similarity(matrix):
	# squared magnitude of preference vectors (number of occurrences)
	square_mag = np.diag(matrix)

	# inverse squared magnitude
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
	matrix = np.dot(items.T, items)
	if cos_similarity:
		matrix = matrix_cos_similarity(matrix)
	if in_perc:
		matrix = matrix_to_perc(matrix)
	if zero_diag:
		matrix = matrix_zero_diag(matrix)
	return matrix

# def cos_similarity(arr, zero_diag, in_perc):
# 	items = np.array(arr)
	
# 	# base similarity matrix (all dot products)
# 	# replace this with items.dot(items.T).todense() for sparse representation
# 	similarity = np.dot(items.T, items)

# 	# squared magnitude of preference vectors (number of occurrences)
# 	square_mag = np.diag(similarity)

# 	# inverse squared magnitude
# 	inv_square_mag = 1 / square_mag

# 	# if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
# 	inv_square_mag[np.isinf(inv_square_mag)] = 0

# 	# inverse of the magnitude
# 	inv_mag = np.sqrt(inv_square_mag)

# 	# cosine similarity (elementwise multiply by inverse magnitudes)
# 	cosine = similarity * inv_mag
# 	cosine = cosine.T * inv_mag

# 	return cosine

	# items = np.array(arr)
	# cooccur_matrix = np.dot(items.T, items)
	# if in_perc:
	# 	diagonal = np.diagonal(cooccur_matrix)
	# 	print(diagonal)
	# 	with np.errstate(divide='ignore', invalid='ignore'):
	# 		cooccur_matrix = np.nan_to_num(np.true_divide(cooccur_matrix, diagonal[:, None]))
	# if zero_diag:
	# 	np.fill_diagonal(cooccur_matrix, 0)
	# return cooccur_matrix
	
	# dataSetI = [0, 1, 1]
	# dataSetII = [1, 1, 0]
	# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

def predict(data, target):	
	switcher = {
		"a": False,
		"b": True,
	}
	
	zero_diag = False
	in_perc = switcher.get(PREDICTION_MODEL, None)
	cos_similarity = True
	
	cooccur = cooccurence(data, cos_similarity, in_perc, zero_diag)
	
	prediction = np.dot(cooccur, target)
	
	if not cos_similarity:
		prediction = np.true_divide(prediction, len(data))
	
	return np.round(np.subtract(prediction, target), 10)

# dataSetI = [0, 1, 1]
# dataSetII = [1, 1, 0]
# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
# print(result)

data = [[1, 0, 1],
        [0, 1, 1],
        [0, 1, 0]]
print("data:")
print(np.array(data))

for i in range(0,3):
	target = [0, 0, 0]
	target[i] = 1

	prediction = predict(data, target)

	print("target:")
	print(np.array(target))

	print("prediction:")
	print(prediction)
