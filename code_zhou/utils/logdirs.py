
import os
from datetime import datetime

def create_log_dict():
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    root = './train_data/' + dt_string
    train_sample_directory = './train_data/' + dt_string + '/sample/'
    model_directory = './train_data/' + dt_string + '/models/'
    generated_model_directory = './train_data/' + dt_string + '/generated_objects/'
    dirs = [root, train_sample_directory, model_directory, generated_model_directory]

    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    return root, train_sample_directory, generated_model_directory, model_directory
