import torch
import os
import pickle
import pandas as pd



"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(base_path):
    if not (os.path.isfile(base_path+'data/wiki-split/processed/data.train') and os.path.isfile(base_path+'data/wiki-split/processed/data.val') and os.path.isfile(base_path+'data/wiki-split/processed/data.test')):
        print('Processing the wiki-split raw data')
        all_s = []
        for split in ['train', 'val', 'test']:
            data_path = base_path + 'data/wiki-split/raw/' + split + '.tsv'
            df = pd.read_csv(data_path, sep='\t', header=None)
            dataset = df.iloc[:, 1].tolist()
            dataset = [d.split('<::::>') for d in dataset]

            for s1, s2 in dataset:
                all_s += [s1 + '\n']
                all_s += [s2 + '\n']

            with open(base_path+'data/wiki-split/processed/data.'+split, 'wb') as f:
                pickle.dump(dataset, f)

        with open(base_path+'data/wiki-split/raw/data.all', 'w') as f:
            f.writelines(all_s)
                

def make_dataset_path(base_path):
    dataset_path = {}
    for split in ['train', 'val', 'test']:
        dataset_path[split] = base_path+'data/wiki-split/processed/data.'+split
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')