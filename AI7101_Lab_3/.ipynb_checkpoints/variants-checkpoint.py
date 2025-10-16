import hashlib
import struct
from itertools import chain


def get_default_tasks_config():
    config = {}
    
    for i in range(17, 18):
        config[f"task{i}"] = 6

    for i in chain(range(3, 5), [6], range(8, 12), range(13, 17), range(18, 19)):
        config[f"task{i}"] = 5
    
    for i in chain(range(1, 2), [7], range(12, 13), range(22, 26)):
        config[f"task{i}"] = 3
    
    for i in chain(range(2, 3), range(19, 20)):
        config[f"task{i}"] = 2
    
    for i in chain(range(5, 6), range(20, 22), range(26, 27)):
        config[f"task{i}"] = 1

    config = dict(sorted(config.items(), key=lambda x: int(x[0][4:])))
    
    return config

def sha256(input_string):
    input_bytes = input_string.encode('utf-8')
    hash_object = hashlib.sha256(input_bytes)
    hash_bytes = hash_object.digest()
    
    return struct.unpack('Q', hash_bytes[:8])[0]

def get_problem(first_name, last_name, task_id, num_tasks):
    unique_string = f"{first_name.lower()}{last_name.lower()}{task_id}"
    
    hash_value = sha256(unique_string)
    
    variant = (hash_value % num_tasks) + 1
    
    return variant

def get_variants_for_all_tasks(first_name, last_name, tasks_config=None):
    if tasks_config is None:
        tasks_config = get_default_tasks_config()

    variants = {}
    for task_id, num_variants in tasks_config.items():
        variants[task_id] = get_problem(first_name, last_name, task_id, num_variants)
    
    return variants