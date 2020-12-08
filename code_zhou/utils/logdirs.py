
import os
from datetime import datetime

def create_log_dict(save_path):
    now = datetime.now()
    if not save_path.endswith('/'):
        save_path+='/'
    save_path+='train_data/'
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    root = save_path + dt_string
    train_sample_directory = root + '/sample/'
    model_directory = root+ '/models/'
    generated_model_directory = root + '/generated_objects/'
    dirs = [root, train_sample_directory, model_directory, generated_model_directory]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return root, train_sample_directory, generated_model_directory, model_directory
