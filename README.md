# Models of Sequential Decision Making

Implementations of basic rl and planning algorithms and domains
mainly for cog sci research.


## Installation

### Installing from GitHub
```bash
$ pip install --upgrade git+https://github.com/markkho/msdm.git
```

### Installing the package in edit mode

After downloading, go into the folder and install the package locally
(with a symlink so its updated as source file changes are made):

```bash
$ pip install -e .
```

It is recommended to use a virtual environment.

Related libraries:
- [BURLAP](https://github.com/jmacglashan/burlap)

## Contributing

To run all tests: `make test`

To run tests for some file: `python -m py.test msdm/tests/$TEST_FILE_NAME.py`

To lint the code: `make lint`
