import numpy as np
from kernel import Kernel
from svm import SVM
from tools.dataset_tools import ACTION_DECODER


def cross_validate(X_train, y_train, X_val, y_val, ker_type="rbf"):
    """
        Cross Validation by predicting over validation_data
        this function is thought for the RBf kernel for now.
        This function saves to a file the log.
    """
    print("Starting Cross Validation Procedure")
    kernel = Kernel(kernel_type=ker_type)
    gammas = [10 ** i for i in range(-3, 0)]
    Cs = [10 ** i for i in range(-3, 0)]

    with open('../resources/cross_val_log.csv', 'a') as f:
        np.savetxt(f, np.array([[ker_type, 'ker_par', 'C', 'accuracy']]), delimiter=";", fmt="%s")

    for g in gammas:
        for c in Cs:
            print(f"\rRunning Cross Validation...", end='')
            svms = {channel_num: SVM(kernel=kernel.kernel, name=f"SVM_ch_{channel_num}")
                    for channel_num in range(X_train.shape[1])}

            for svm_num in list(svms.keys()):
                svms[svm_num].fit(np.array(X_train[:, svm_num, :]), np.array(y_train))

            accu = test_svms(svms, X_val, y_val)
            with open('../resources/cross_val_log.csv', 'a') as f:
                np.savetxt(f, np.array([[g, c, accu]]), delimiter=";", fmt="%s")


def maj_prediction(svms, xi):
    predictions = []
    for svm_num in list(svms.keys()):
        predictions.append(svms[svm_num].predict(xi[svm_num]))
    ones = sum([1 for p in predictions if p == 1])
    if ones > len(predictions) / 2:
        confidence = ones / len(predictions)
        ret = 1
    elif ones < len(predictions) / 2:
        confidence = (len(predictions) - ones) / len(predictions)
        ret = -1
    else:
        ret = [-1, 1][np.random.randint(0, 2)]
        confidence = 0.5
    return ret, confidence


def test_svms(svms, X_test, y_test, verbose=False):
    """
        Uses the given svms to predict the output, each
        svm will be fit to a particular channel, if most of the
        channel will turn the svms over a certain label then
        that will be the prediction, i.e. it perform majority voting.
    :param svms: svms to fit
    :param X_test: testing data
    :param y_test: testing labels
    :param verbose: parameter in order to have prints for individual predictions
    :return accuracy of the overall system of svms
    """

    y_pred = []
    for i, el in enumerate(X_test):
        y_pred_i, conf = maj_prediction(svms, np.array(el))
        y_pred.append(y_pred_i)
        if verbose:
            print(f"{ACTION_DECODER[y_pred_i]} should be: "
                  f"{ACTION_DECODER[y_test[i]]}, confidence: {conf}")

    return SVM.accuracy(y_pred, y_test)
