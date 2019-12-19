import functionsBDT
from joblib import load

BASE_PATH = "/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/"

def main():
    print("Optimising BDT...")

    save_path =  BASE_PATH + 'ml_calculated_data/equal_weight/'

    X_train_new = load(save_path + 'bdt_X_train_new.joblib')
    y_train = load(save_path + 'bdt_y_train.joblib')
    X_test_new = load(save_path + 'bdt_X_test_new.joblib')
    y_test = load(save_path + 'bdt_y_test.joblib')

    functionsBDT.optimise_bdt(X_train_new, y_train, X_test_new, y_test)


if __name__ == "__main__":
    main()