import json

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def model_training(args):

    # Load and unpack the train_data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    x_train, x_test, y_train, y_test  = data 
    
    # Initialize and train the model
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)

    # Get predictions
    y_pred = model.predict(x_test)
    
    # Get accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save output into file
    with open(args.accuracy, 'w') as accuracy_file:
        accuracy_file.write(str(accuracy))



if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--accuracy', type=str)

    args = parser.parse_args()

    # Creating the directory where the output file will be created (the directory may or may not exist).
    Path(args.accuracy).parent.mkdir(parents=True, exist_ok=True)
    
    model_training(args)
