from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='msdm',
      version='0.4',
      description='Models of sequential decision making',
      url='https://github.com/markkho/msdm',
      author="Mark Ho",
      author_email='mark.ho.cs@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'matplotlib', 'scipy', 'torch', 'tqdm', 'sparse', 'termcolor',
          'frozendict @ git+https://github.com/Marco-Sulla/python-frozendict@v2.0.1'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
