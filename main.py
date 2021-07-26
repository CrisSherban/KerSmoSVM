import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernel import Kernel
from svm import SVM
from test_svm import TestSVM
from tools.dataset_tools import preprocess_raw_eeg, eeg_plotter, load_data, split_data
from tools.validation_tools import test_svms, cross_validate


def main():
    # one can run the unittest by itself by running
    # python test_svm.py
    # in the project main directory
    TestSVM().run()

    # splits dataset, uncomment and use it once
    # delete training, testing and validation folder before rerun
    # split_data(splitting_percentage=(80, 10, 10))

    # loading personal_dataset
    X_train_tmp, y_train = load_data(starting_dir="resources/training_data", shuffle=False, balance=True)
    X_val_tmp, y_val = load_data(starting_dir="resources/validation_data", shuffle=False, balance=True)
    X_test_tmp, y_test = load_data(starting_dir="resources/test_data", shuffle=False, balance=True)

    # cleaning the raw personal_dataset
    X_train, X_train_fft = preprocess_raw_eeg(X_train_tmp, lowcut=7, highcut=20, coi3order=0)
    X_val, X_val_fft = preprocess_raw_eeg(X_val_tmp, lowcut=7, highcut=20, coi3order=0)
    X_test, X_test_fft = preprocess_raw_eeg(X_test_tmp, lowcut=7, highcut=20, coi3order=0)

    # let's take a look
    print("EEG dataset shape of X_train: ", X_train.shape)
    eeg_plotter(X_train[0], y_train[0])
    # cross_validate(X_train, y_train, X_val, y_val)

    # create an SVM for each channel and take the majority vote
    # while looping through a set of different kernels
    accus = []
    kernels = ['linear', 'poly', 'rbf']
    for kernel_type in kernels:
        kernel = Kernel(kernel_type=kernel_type)
        svms = {channel_num: SVM(kernel=kernel.kernel, name=f"SVM_ch_{channel_num}", verbose=False)
                for channel_num in range(X_train.shape[1])}

        print(f"Fitting SVMs with kernel: {kernel}")
        for svm_num in list(svms.keys()):
            svms[svm_num].fit(np.array(X_train[:, svm_num, :]), np.array(y_train))

        # still using X_val and y_val since this process can
        # be included in the tuning of the model hyperparameters
        # which in this case are the kernels
        accu = test_svms(svms, X_val, y_val, verbose=False)
        print(f"Accuracy of 8 SVMs (one for each channel): {accu}")
        accus.append([kernel_type, accu])

    print("Close the figure to continue")
    bars = pd.DataFrame(accus, columns=['Kernel', 'Accuracy'])
    bars.plot(x='Kernel', y=['Accuracy'], kind="bar")
    plt.tight_layout()
    # plt.savefig('pictures/kernel_comparison.png')
    plt.show()


if __name__ == '__main__':
    main()
