import ROOT
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

from root_numpy import root2array, rec2array, array2root
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from joblib import dump, load

from xgboost import XGBClassifier
from sklearn import model_selection
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from matplotlib.legend_handler import HandlerLine2D

BASE_PATH = "/unix/nemo3/users/ldawson/"
NEW_EXPORT_PATH = "/unix/nemo3/users/ldawson/equalWeights"
# Limit what we extract from the ROOT files.
BRANCH_NAMES_TRAIN = """reco.total_calorimeter_energy, reco.higher_electron_energy,
reco.lower_electron_energy, reco.angle_between_tracks, reco.internal_probability,
reco.external_probability, weights""".split(",")
BRANCH_NAMES_TRAIN = [c.strip() for c in BRANCH_NAMES_TRAIN]

BRANCH_NAMES_TEST = """reco.total_calorimeter_energy, reco.higher_electron_energy,
reco.lower_electron_energy, reco.angle_between_tracks, reco.internal_probability,
reco.external_probability""".split(",")
BRANCH_NAMES_TEST = [c.strip() for c in BRANCH_NAMES_TEST]
SMALL_DATA = False

def import_data_small():
    signal = root2array(BASE_PATH + "sensitivity_0nubb_1E5_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    signal = rec2array(signal)

    bkg2nu = root2array(BASE_PATH + "sensitivity_2nubb_1E5_Small_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg2nu = rec2array(bkg2nu)

    bkg214Bi = root2array(BASE_PATH + "sensitivity_Bi214_Foils_Small_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg214Bi = rec2array(bkg214Bi)

    bkg208Tl = root2array(BASE_PATH + "sensitivity_Tl208_Foils_Small_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg208Tl = rec2array(bkg208Tl)

    bkgRn = root2array(BASE_PATH + "sensitivity_Bi214_Wires_Small_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkgRn = rec2array(bkgRn)

    return signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn

def import_data_small_2():
    signal = root2array(BASE_PATH + "sensitivity_0nubb_1E5_Pred_With_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TEST)
    signal = rec2array(signal)

    bkg2nu = root2array(BASE_PATH + "sensitivity_2nubb_1E5_Small_Pred_With_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TEST)
    bkg2nu = rec2array(bkg2nu)

    bkg214Bi = root2array(BASE_PATH + "sensitivity_Bi214_Foils_Small_Pred_With_Cut.root",
                          "Sensitivity",
                          BRANCH_NAMES_TEST)
    bkg214Bi = rec2array(bkg214Bi)

    bkg208Tl = root2array(BASE_PATH + "sensitivity_Tl208_Foils_Small_Pred_With_Cut.root",
                          "Sensitivity",
                          BRANCH_NAMES_TEST)
    bkg208Tl = rec2array(bkg208Tl)

    bkgRn = root2array(BASE_PATH + "sensitivity_Bi214_Wires_Small_Pred_With_Cut.root",
                       "Sensitivity",
                       BRANCH_NAMES_TEST)
    bkgRn = rec2array(bkgRn)

    return signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn

def import_data():
    signal = root2array(BASE_PATH + "Signal/0nubb/MachineLearning/sensitivity_0nubb_1E7_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    signal = rec2array(signal)

    bkg2nu = root2array(BASE_PATH + "Signal/2nubb/MachineLearning/sensitivity_2nubb_2E8_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg2nu = rec2array(bkg2nu)

    bkg214Bi = root2array(BASE_PATH + "Backgrounds/Bi214_Foils/MachineLearning/sensitivity_Bi214_Foils_2E8_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg214Bi = rec2array(bkg214Bi)

    bkg208Tl = root2array(BASE_PATH + "Backgrounds/Tl208_Foils/MachineLearning/sensitivity_Tl208_Foils_2E8_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkg208Tl = rec2array(bkg208Tl)

    bkgRn = root2array(BASE_PATH + "Backgrounds/Bi214_Wires/MachineLearning/sensitivity_Bi214_Wires_2E8_Pre_Cut.root",
                        "Sensitivity",
                        BRANCH_NAMES_TRAIN)
    bkgRn = rec2array(bkgRn)

    return signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn

def import_data_2():
    print("Loading 0nubb signal to structured array...")
    signal = root2array(BASE_PATH + "Signal/0nubb/MachineLearning/sensitivity_0nubb_1E7_Pre_Cut_2.root",
                        "Sensitivity",
                        BRANCH_NAMES_TEST)
    print("Convert 0nubb signal structured array to ndarray array...")
    signal = rec2array(signal)

    print("Loading 2nubb bkg2nu to structured array...") 
    bkg2nu = root2array(BASE_PATH + "Signal/2nubb/MachineLearning/sensitivity_2nubb_2E8_Pre_Cut_2.root",
                        "Sensitivity",
                        BRANCH_NAMES_TEST)
    print("Convert 2nubb bkg2nu structured array to ndarray array...")
    bkg2nu = rec2array(bkg2nu)

    print("Loading Bi214 bkg214Bi to structured array...") 
    bkg214Bi = root2array(BASE_PATH + "Backgrounds/Bi214_Foils/MachineLearning/sensitivity_Bi214_Foils_2E8_Pre_Cut_2.root",
                          "Sensitivity",
                          BRANCH_NAMES_TEST)
    print("Convert Bi214 bkg214Bi structured array to ndarray...")
    bkg214Bi = rec2array(bkg214Bi)

    print("Loading Tl208 bkg208T1 to structured array...") 
    bkg208Tl = root2array(BASE_PATH + "Backgrounds/Tl208_Foils/MachineLearning/sensitivity_Tl208_Foils_2E8_Pre_Cut_2.root",
                          "Sensitivity",
                          BRANCH_NAMES_TEST)
    print("Convert Tl208 bkg208T1 structured array to ndarray...")
    bkg208Tl = rec2array(bkg208Tl)

    print("Loading Radon bkgRn to structured array to ndarray...") 
    bkgRn = root2array(BASE_PATH + "Backgrounds/Bi214_Wires/MachineLearning/sensitivity_Bi214_Wires_2E8_Pre_Cut_2.root",
                       "Sensitivity",
                       BRANCH_NAMES_TEST)
    print("Convert Radon bkgRn structured array to ndarray...")
    bkgRn = rec2array(bkgRn)

    return signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn

def compare_train_test(clf, X_train_new, y_train, X_test_new, y_test, bins=30):
    decisions = []
    for X,y in ((X_train_new, y_train), (X_test_new, y_test)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)

    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    # width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig(BASE_PATH + 'Plots/compare_train_test.png')
    #plt.show()

def plot_roc_curve(bdt, X_test_new, y_test):
    decisions = bdt.decision_function(X_test_new)
    fpr, tpr, _ = roc_curve(y_test, decisions)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(BASE_PATH + 'Plots/roc_equal_weight.png')
    
    #plt.show()

def train_bdt():
    print("Loading data...")
    if SMALL_DATA:
        signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn = import_data_small()
    else:
        signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn = import_data()

    # print("Sampling 10% of the data for training")
    # #Create smaller samples, 10% of the size
    signal = np.asarray(random.sample(signal, int((len(signal))*0.1)))
    bkg2nu = np.asarray(random.sample(bkg2nu, int((len(bkg2nu))*0.1)))
    bkg214Bi = np.asarray(random.sample(bkg214Bi, int((len(bkg214Bi))*0.1)))
    bkg208Tl = np.asarray(random.sample(bkg208Tl, int((len(bkg208Tl))*0.1)))
    bkgRn = np.asarray(random.sample(bkgRn, int((len(bkgRn))*0.1)))

    print("Creating arrays...")
    # X = Features (i.e. the data)
    X = np.concatenate((signal,
                       bkg2nu,
                       bkg214Bi,
                       bkg208Tl,
                       bkgRn))

    # y = Labels (i.e. what it is, signal / background)
    y = np.concatenate((np.ones(signal.shape[0]),
                       np.zeros(bkg2nu.shape[0]),
                       np.zeros(bkg214Bi.shape[0]),
                       np.zeros(bkg208Tl.shape[0]),
                       np.zeros(bkgRn.shape[0])))

    print("Splitting Data...")
    # Split the data
    X_dev,X_eval,y_dev,y_eval = train_test_split(X,
                                                 y,
                                                 test_size=0.33,
                                                 random_state=48)

    X_train,X_test,y_train,y_test = train_test_split(X,
                                                     y,
                                                     test_size=0.33,
                                                     random_state=42)

    # print("Oversampling...")
    # # Oversample to improve representation of backgrounds
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    # X_test_resampled, y_test_resampled = ros.fit_sample(X_test, y_test)
    # X_dev_resampled, y_dev_resampled = ros.fit_sample(X_dev, y_dev)
    # X_eval_resampled, y_eval_resampled = ros.fit_sample(X_eval, y_eval)
    # print(sorted(Counter(y_resampled).items()))

    print("Removing weights..")
    # Remove weights on backgrounds (will be passed in to the BDT later)
    # 30/09/19 - removed re sampling
    X_train_weights = X_train[:,6]
    X_train_new = np.delete(X_train,6,axis=1)
    X_test_new = np.delete(X_test,6,axis=1)

    X_dev_weights = X_dev[:,6]
    X_dev_new = np.delete(X_dev,6,axis=1)
    X_eval_new = np.delete(X_eval,6,axis=1)

    print("Creating classifier for DT")
    # Create classifiers
    dt = DecisionTreeClassifier(max_depth=12,
                                min_samples_split=0.5,
                                min_samples_leaf=400)

    print("Creating classifier for BDT")
    bdt = AdaBoostClassifier(dt,
                            algorithm='SAMME',
                            n_estimators=1200,
                            learning_rate=0.5)

    print("Fitting BDT...")
    # Train the classifier - pass in weights from earlier
    fitted_tree = bdt.fit(X_train_new, y_train)#, sample_weight=X_train_weights)

    print("Predicting on training data...")
    # Use the fitted tree to predict on training data and new test data
    y_predicted_train = bdt.predict(X_train_new)

    print("Predicting on test data...")
    y_predicted_test = bdt.predict(X_test_new)

    print(classification_report(y_train, y_predicted_train, target_names=["signal", "background"]))
    print("Area under ROC curve for training data: {0:.4f}".format(roc_auc_score(y_train,
                                                                                 bdt.decision_function(X_train_new))))

    print(classification_report(y_test, y_predicted_test, target_names=["signal", "background"]))
    print("Area under ROC curve for test data: {0:.4f}".format(roc_auc_score(y_test,
                                                                             bdt.decision_function(X_test_new))))

    plot_roc_curve(bdt, X_test_new, y_test)
    compare_train_test(bdt, X_train_new, y_train, X_test_new, y_test)

    print("Saving classifier...")
    save_path =  BASE_PATH + 'ml_calculated_data/equal_weight/'
    dump(bdt, save_path + 'bdt_classifier.joblib')
    dump(fitted_tree, save_path + 'bdt_fitted_tree.joblib')
    dump(X_train_new, save_path + 'bdt_X_train_new.joblib')
    dump(X_test_new, save_path + 'bdt_X_test_new.joblib')
    dump(X_dev_new, save_path + 'bdt_X_dev_new.joblib')
    dump(X_dev_weights, save_path + 'bdt_X_dev_weights.joblib')
    dump(X_eval_new, save_path + 'bdt_X_eval_new.joblib')
    dump(y_test, save_path + 'bdt_y_test.joblib')
    dump(y_train, save_path + 'bdt_y_train.joblib')
    dump(y_dev, save_path + 'bdt_y_dev.joblib')
    dump(y_eval, save_path + 'bdt_y_eval.joblib')

    print("Finished Training.")

def import_classifier():
    print("Importing classifier data...")
    
    # return bdt, fitted_tree, X_train_new, X_test_new, X_dev_new, X_dev_weights, X_eval_new, y_test_resampled, y_resampled, y_dev_resampled, y_eval_resampled

def predict(bdt):

    print("Starting predict function...")
    print("Importing Data 2...")
    if SMALL_DATA:
        signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn = import_data_small_2()
    else:
        signal, bkg2nu, bkg214Bi, bkg208Tl, bkgRn = import_data_2()

    #Create smaller samples, 1% of the size
    print("Predicting on a sample of 1% of the data")
    signal = np.asarray(random.sample(signal, int((len(signal))*0.01)))
    bkg2nu = np.asarray(random.sample(bkg2nu, int((len(bkg2nu))*0.01)))
    bkg214Bi = np.asarray(random.sample(bkg214Bi, int((len(bkg214Bi))*0.01)))
    bkg208Tl = np.asarray(random.sample(bkg208Tl, int((len(bkg208Tl))*0.01)))
    bkgRn = np.asarray(random.sample(bkgRn, int((len(bkgRn))*0.01)))

    # print("Concatenating arrays...")
    # X = np.concatenate((signal,
    #                    bkg2nu,
    #                    bkg214Bi,
    #                    bkg208Tl,
    #                    bkgRn))
    # y = np.concatenate((np.ones(signal.shape[0]),
    #                    np.zeros(bkg2nu.shape[0]),
    #                    np.zeros(bkg214Bi.shape[0]),
    #                    np.zeros(bkg208Tl.shape[0]),
    #                    np.zeros(bkgRn.shape[0])))

    # print("Predicting Values...")
    # y_predicted = bdt.decision_function(X)
    # y_predicted.dtype = [('y', np.float64)]
    # print(bdt.feature_importances_)

    y_pred_signal = bdt.decision_function(signal)
    y_pred_2nu = bdt.decision_function(bkg2nu)
    y_pred_Bi = bdt.decision_function(bkg214Bi)
    y_pred_Tl = bdt.decision_function(bkg208Tl)
    y_pred_Rn = bdt.decision_function(bkgRn)

    y_pred_signal.dtype = [('y', np.float64)]
    y_pred_2nu.dtype = [('y', np.float64)]
    y_pred_Bi.dtype = [('y', np.float64)]
    y_pred_Tl.dtype = [('y', np.float64)]
    y_pred_Rn.dtype = [('y', np.float64)]

    decay_modes = {
        "0nu": {
            "existing": BASE_PATH + "0nubb/sensitivity_0nubb_1E7_Pre_Cut_2.root",
            "new": NEW_EXPORT_PATH + "test-pred-0nu-1E7.root",
            "pred": y_pred_signal
        },
        "2nu": {
            "existing": BASE_PATH + "2nubb/sensitivity_2nubb_2E8_Pre_Cut_2.root",
            "new": NEW_EXPORT_PATH + "test-pred-2nu-2E8.root",
            "pred": y_pred_2nu
        },
        "bifoil": {
            "existing": BASE_PATH + "Bi214/sensitivity_Bi214_Foils_2E8_Pre_Cut_2.root",
            "new": NEW_EXPORT_PATH + "test-pred-bifoil-2E8.root",
            "pred": y_pred_Bi
        },
        "tlfoil": {
            "existing": BASE_PATH + "Tl208/sensitivity_Tl208_Foils_2E8_Pre_Cut_2.root",
            "new": NEW_EXPORT_PATH + "test-pred-tlfoil-2E8.root",
            "pred": y_pred_Tl
        },
        "rn": {
            "existing": BASE_PATH + "Radon/sensitivity_Bi214_Wires_2E8_Pre_Cut_2.root",
            "new": NEW_EXPORT_PATH + "test-pred-radon-2E8.root",
            "pred": y_pred_Rn
        }
    }

    print("Write data to root files...")
    for _, values in decay_modes.items():
        array2root(values["pred"], values["new"], "BDToutput", "recreate")

    # print("Write data to root files...")
    # total = 0
    # for _, values in decay_modes.items():
    #     data_length = len(values['data'])
    #     print(data_length)
    #     print(y_predicted[total])
    #     pred_value = y_predicted[total:total + data_length]
    #     total += data_length

    #     # new_column = np.array(pred_value, dtype=[('bdt', 'float')])
    #     # array2root(new_column, values["existing"], 'Sensitivity')
    #     array2root(pred_value, values["new"], "BDToutput", "recreate")
        

def validate(bdt, X_dev_new, y_dev, X_dev_weights, y_eval, X_eval_new):
    gbt = GradientBoostingClassifier(n_estimators=200,
                                     max_depth=1,
                                     subsample=0.5,
                                     max_features=0.5,
                                     learning_rate=0.02)

    xgb = XGBClassifier()
    param_grid = {'n_estimators': [50, 100, 500, 1000, 1200],
                  'learning_rate': [0.1, 0.2, 0.5]}

    clf = model_selection.GridSearchCV(bdt,
                                       param_grid,
                                       cv=3,
                                       scoring='roc_auc',
                                       n_jobs=8)
    clf.fit(X_dev_new,
            y_dev,
            sample_weight=X_dev_weights)

    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))

    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    y_true, y_pred = y_eval, clf.predict(X_eval_new)
    print(classification_report(y_true, y_pred))

def optimise_bdt(X_train_new, y_train, X_test_new, y_test):
    """
    The first parameter to tune is max_depth. This indicates how deep the tree
    can be. The deeper the tree, the more splits it has and it captures more
    information about the data. We fit a decision tree with depths ranging from 1
    to 32 and plot the training and test auc scores.

    min_samples_split represents the minimum number of samples required to split
    an internal node. This can vary between considering at least one sample at
    each node to considering all of the samples at each node. When we increase
    this parameter, the tree becomes more constrained as it has to consider more
    samples at each node. Here we will vary the parameter from 10% to 100% of the samples

    min_samples_leaf is The minimum number of samples required to be at a leaf
    node. This parameter is similar to min_samples_splits, however, this describe
    the minimum number of samples of samples at the leafs, the base of the tree.
    """
    optimisation_variables_properties = {
        "max_depth": {
            "start": 1,
            "stop": 32,
            "num": 32,
            "dtype": None
        },

        "min_samples_split": {
            "start": 0.1,
            "stop": 1.0,
            "num": 10,
            "dtype": None
        },
        "min_samples_leaf": {
            "start": 1,
            "stop": 2000,
            "num": 100,
            "dtype": int
        }
    }

    for key, value in optimisation_variables_properties.items():
        optimisation_variables = np.linspace(value['start'],
                                             value['stop'],
                                             value['num'],
                                             endpoint=True,
                                             dtype=value['dtype'])
        train_results = []
        test_results = []

        for optimisation_variable in optimisation_variables:
            if key == "max_depth":
                dt = DecisionTreeClassifier(max_depth=optimisation_variable)
            elif key == "min_samples_split":
                dt = DecisionTreeClassifier(min_samples_split=optimisation_variable)
            elif key == "min_samples_leaf":
                dt = DecisionTreeClassifier(min_samples_leaf=optimisation_variable)

            dt.fit(X_train_new, y_train)
            train_pred = dt.predict(X_train_new)
            false_positive_rate, true_positive_rate, _ = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            y_pred = dt.predict(X_test_new)
            false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            test_results.append(roc_auc)

        line1, = plt.plot(optimisation_variables, train_results, 'b', label="Train AUC")
        plt.plot(optimisation_variables, test_results, 'r', label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel(key)
        plt.savefig(BASE_PATH + 'Plots/optimise_bdt.png')
        #plt.show()
