#!/usr/bin/env python

from __future__ import print_function
import numpy as np

PREDICTION_MODEL = "a"

def cooccurence(arr, zero_diag, in_perc):
	items = np.array(arr)
	cooccur_matrix = np.dot(items.T, items)
	if in_perc:
		diagonal = np.diagonal(cooccur_matrix)
		print(diagonal)
		with np.errstate(divide='ignore', invalid='ignore'):
			cooccur_matrix = np.nan_to_num(np.true_divide(cooccur_matrix, diagonal[:, None]))
	if zero_diag:
		np.fill_diagonal(cooccur_matrix, 0)
	return cooccur_matrix

def predict(data, target):
	switcher = {
		"a": False,
		"b": True,
	}
	
	zero_diag = True
	in_perc = switcher.get(PREDICTION_MODEL, None)
	
	cooccur = cooccurence(data, zero_diag, in_perc)
	print("cooccur_matrix:")
	print(cooccur)
	
	prediction = np.dot(cooccur, target)
	return prediction
	print("p1:")
	print(prediction)
	prediction = np.true_divide(prediction, len(data))
	print("p2:")
	print(prediction)
	return np.subtract(prediction, target)

data = [[1, 0, 1, 1],
        [1, 1, 1, 1],
        [1, 0, 1, 1]]
print("data:")
print(np.array(data))

target = [1, 0, 1, 1]

prediction = predict(data, target)

print("target:")
print(np.array(target))

print("prediction:")
print(prediction)
