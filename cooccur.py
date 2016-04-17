#!/usr/bin/env python

# http://stackoverflow.com/questions/20574257/constructing-a-co-occurrence-matrix-in-python-pandas

from __future__ import print_function
import numpy as np
np.random.seed(3) # for reproducibility

def get_coll_titles():
	return 'Alice Bob Carol'.split(' ')

def get_num_items():
	return 4

def example1():
	# Generate data: 5 labels, 10 examples, binary.
	people = get_coll_titles()
	num_people = len(people)
	num_items = get_num_items()

	# binary here but could be any integer.
	data = np.random.randint(0, 2, (num_people, num_items))

	for header, series in zip(people, data):
		print("{0:5} {1}".format(header, series))
	print()

	# Compute cooccurrence matrix
	cooccur_matrix = np.dot(data.T, data)

	print("cooccur_matrix:")
	for data in cooccur_matrix:
		print("{0}".format(data))
	print()

	# Compute cooccurrence matrix in percentage
	# FYI: http://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element
	#      http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero/32106804#32106804
	diagonal = np.diagonal(cooccur_matrix)
	with np.errstate(divide='ignore', invalid='ignore'):
		cooccur_matrix_perc = np.nan_to_num(np.true_divide(cooccur_matrix, diagonal[:, None]))

	print("cooccur_matrix_perc:")
	for data in cooccur_matrix_perc:
		print("[", end="")
		for cell in data:
			print("{0:4.0f}% ".format(cell * 100), end="")
		print("]")
	print()

def example2():
	people = 'Snack Trans Dop'.split(' ')
	num_people = len(people)
	item_header = ['AA', 'SL', 'BB', 'D0', 'Dk', 'FF'];
	items = np.array([['1', '0', '1', '1', '0', '0'],
			 ['1', '1', '1', '0', '0', '1'],
			 ['1', '0', '1', '0', '1', '1']])
	num_items = len(item_header)

	print(item_header)
	print(items)
	print()

	# Compute cooccurrence matrix
	items_asint = items.astype(int)
	coocc = items_asint.T.dot(items_asint)
	print(coocc)
	print()

	np.fill_diagonal(coocc, 0)
	print(coocc)
	print()

example1()
example2()