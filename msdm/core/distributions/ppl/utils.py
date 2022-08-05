import re
from msdm.core.distributions.dictdistribution import DictDistribution

def strip_comments(code):
    code = str(code)
    return re.sub(r'(?m)^ *#.*\n?', '', code)

def flip(p):
    return DictDistribution({True: p, False: 1 - p})
