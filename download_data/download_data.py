import pickle

import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def download_data(args):

    # Gets and split dataset
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    data = x_train, x_test, y_train, y_test 
        
    #Save the train_data and test_data as a pickle file to be used by the next component.
    with open((args.data, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    
    # This component does not receive any input
    # it only outpus one artifact which is `data`.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    
    args = parser.parse_args()
    
    # Creating the directory where the output file will be created 
    # (the directory may or may not exist).
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)

    download_data(args)
    
