import numpy as np
import json


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


"""
Make dataset from files refered by indice
Input: any list of indice from 0 to 29, 
Output: Trainable Tensor dataset (data, label)
"""
def load_dataset(idx_list, mode='train', path='.'): 
    assert type(idx_list) is list
    assert mode in ['train', 'test']
    
    datasets_x = []
    datasets_y = []
    
    # Flatten data for each json file
    for idx in idx_list:
        user_data = load_json(f'{path}/{mode}/all_data_{idx}_niid_0_keep_0_train_9.json')['user_data']
        for folder in user_data.keys():
            for data in user_data[folder]['x'][1:]:
                datasets_x.append(np.array(data, dtype='float64').reshape((28, 28)))
            for data in user_data[folder]['y'][1:]:
                datasets_y.append(np.array(data, dtype='uint8'))
    
    return (datasets_x, datasets_y)