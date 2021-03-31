test:
	python -m py.test

lint:
	# Copied from .github/workflows
	flake8 msdm --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 msdm --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
