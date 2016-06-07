.PHONY: all test

all: test

test:
	@./db.py test.db useritems
