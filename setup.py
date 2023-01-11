from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='msdm',
    version='0.10',
    description='Models of sequential decision making',
    keywords = [
        'reinforcement learning',
        'planning',
        'cognitive science'
    ],
    url='https://github.com/markkho/msdm',
    download_url = 'https://github.com/markkho/msdm/archive/refs/tags/v0.10.tar.gz',
    author="Mark Ho",
    author_email='mark.ho.cs@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'tqdm',
        'pandas',
        'frozendict',
        'termcolor',
        'julia',
        'cvxpy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
