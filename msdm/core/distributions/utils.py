
import hashlib

def obj_seed(obj):
    return int(hashlib.sha1(hash(obj).to_bytes(8, 'big', signed=True)).hexdigest(), 16)