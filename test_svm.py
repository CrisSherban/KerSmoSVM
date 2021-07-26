import unittest
from kernel import Kernel
from svm import SVM
import numpy as np


class TestSVM(unittest.TestCase):
    def run(self, kernel="rbf"):
        """
            Just a testing function to assert the usage of the SVM
            while classifying 0 or 1 handwritten digits.
            This function uses only a portion of the original dataset to make testing faster.
        :param kernel: str, kernel type
        """

        print("------------")
        print("Testing an SVM on MNIST dataset over binary classification of 0 and 1")
        mnist = np.loadtxt("resources/mnist_reduced/mnist_reduced.csv", delimiter=',')
        lbls, data = mnist[:, 0], mnist[:, 1:]
        X_train, y_train, X_test, y_test = data[:800], lbls[:800], data[800:], lbls[800:]

        # selecting only 1 and 0 since the SVM class is a binary classifier
        idxs_train = np.where((y_train == 0) | (y_train == 1))[0]
        idxs_test = np.where((y_test == 0) | (y_test == 1))[0]
        f = lambda x: x if x == 1 else -1

        # converting 0 to -1
        y_train = np.array([f(i) for i in y_train[idxs_train]])
        y_test = np.array([f(i) for i in y_test[idxs_test]])

        # normalize
        X_train = X_train[idxs_train] / 255
        X_test = X_test[idxs_test] / 255

        # instantiate kernel and svm and test
        kernel = Kernel(kernel_type=kernel)
        svm = SVM(kernel=kernel.kernel)
        print(svm)
        print(f"Fitting SVM")
        svm.fit(np.array(X_train), np.array(y_train))

        # eye check
        y_pred = []
        for i, el in enumerate(X_test):
            y_pred.append(svm.predict(np.array(el)))
        accuracy = SVM.accuracy(y_pred, y_test)
        print(f"Accuracy on TEST SET over MNIST: {accuracy:.2F}")

        # since this problem can be solved easily with SVMs
        # reaching 0.8 accuracy will be our test
        self.assertGreater(accuracy, 0.8)
        print("Done testing on MNIST")
        print("------------")


if __name__ == '__main__':
    unittest.main()
