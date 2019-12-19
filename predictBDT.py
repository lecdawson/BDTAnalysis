import functionsBDT
from joblib import load

BASE_PATH = "/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/"

def loadTree():
    print("Loading classifier...")
    save_path =  BASE_PATH + 'ml_calculated_data/equal_weight/'

    return load(save_path + 'bdt_classifier.joblib')

def main():
    print("Predicting with BDT...")

    functionsBDT.predict(loadTree())


if __name__ == "__main__":
    main()
