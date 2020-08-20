from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pyrlap',
      version='0.4',
      description='Library for combining MDPs and solving them for cognitive '+
                  'science research',
      url='https://github.com/markkho/pyrlap',
      author="Mark Ho",
      author_email='mark.ho.cs@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'scipy', 'torch', 'tqdm'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
