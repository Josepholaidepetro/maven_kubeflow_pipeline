
def print_result(args):
    # Print results
    with open(args.accuracy, 'r') as f:
        score = f.read()

    print(f"Random forest (accuracy): {score}")
    
if __name__ == '__main__':

    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--data', type=str)
    parser.add_argument('--accuracy', type=str)

    args = parser.parse_args()
    
    model_training(args)
