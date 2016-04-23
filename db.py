#!/usr/bin/env python

import sys
import sqlite3
import cooccur

def main():
	try:
		db_name = sys.argv[1]
	except IndexError:
		print "No DB name provided!\n./db.py [db]"
		sys.exit(1)

	print db_name

if __name__ == "__main__":
	main()