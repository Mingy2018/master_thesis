
import os

def create_log_dict(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return 0
