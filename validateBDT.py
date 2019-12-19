import functionsBDT
from joblib import load

BASE_PATH = "/home/vagrant/MachineLearning/SciKit-Learn/rootFiles/"

def main():
    print("Validating BDT...")

    save_path =  BASE_PATH + 'ml_calculated_data/equal_weight/'

    bdt = load(save_path + 'bdt_classifier.joblib')
    X_dev_new = load(save_path + 'bdt_X_dev_new.joblib')
    y_dev = load(save_path + 'bdt_y_dev.joblib')
    X_dev_weights = load(save_path + 'bdt_X_dev_weights.joblib')
    y_eval = load(save_path + 'bdt_y_eval.joblib')
    X_eval_new = load(save_path + 'bdt_X_eval_new.joblib')

    functionsBDT.validate(bdt, X_dev_new, y_dev, X_dev_weights, y_eval, X_eval_new)


if __name__ == "__main__":
    main()