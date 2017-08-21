import glob
import pickle
import time
import cv2
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

from featurizer import extract_features


class CarClassifier:
    def __init__(self,
                 colorspace='YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                 orient=9,
                 pix_per_cell=8,
                 cell_per_block=2,
                 hog_channel='ALL',  # Can be 0, 1, 2, or "ALL"
                 kernel = 'linear'
                 ):
        self.colorspace = colorspace  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = int(hog_channel)
        self.clf = None
        self.scaler = None
        self.kernel = kernel

    def fit(self, images, random_state):
        # Divide up into cars and notcars
        cars = []
        notcars = []
        for image in images:
            if 'Extras' in image:
                notcars.append(mpimg.imread(image))
            elif 'GTI_Right' in image:
                cars.append(mpimg.imread(image))
            elif 'GTI_Left' in image:
                cars.append(cv2.flip(mpimg.imread(image), 1))

        # balance classes by subsampling negative classes
        np.random.shuffle(notcars)
        notcars = notcars[0:int(len(cars) * 1.5)]
        _notcars = [
            mpimg.imread('data/non-vehicles/Extras/extra955.png'),
            mpimg.imread('data/non-vehicles/Extras/extra956.png'),
            mpimg.imread('data/non-vehicles/Extras/extra957.png'),
            mpimg.imread('data/non-vehicles/Extras/extra958.png'),
            mpimg.imread('data/non-vehicles/Extras/extra997.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1000.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1001.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1005.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1008.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1009.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1010.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1011.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1012.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1013.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1014.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1015.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1058.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1061.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1062.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1063.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1064.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1065.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1066.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1067.png'),
            mpimg.imread('data/non-vehicles/Extras/extra1068.png')]
        _notcars_flipped = [cv2.flip(x, 1) for x in _notcars]
        notcars += _notcars + _notcars + _notcars + _notcars + \
                   _notcars_flipped + _notcars_flipped + _notcars_flipped + _notcars_flipped

        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time
        # sample_size = 500
        # cars = cars[0:sample_size]
        # notcars = notcars[0:sample_size]

        print("Number of postive classes: {}".format(len(cars)))
        print("Number of negtaive classes: {}".format(len(notcars)))

        t = time.time()
        car_features = \
            extract_features(cars, cspace=self.colorspace, orient=self.orient,
                             pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                             hog_channel=self.hog_channel, feature_vec=True)
        notcar_features = \
            extract_features(notcars, cspace=self.colorspace, orient=self.orient,
                             pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                             hog_channel=self.hog_channel, feature_vec=True)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        print('Normalizing feature vectors...')
        scaled_X = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        print('Splitting into training and testing sets...')
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=random_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use an SVC
        svc = SVC()
        # Cross-validation with grid search
        # parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 5]}
        parameters = {'kernel': [self.kernel], 'C': [2, 3]}
        self.clf = GridSearchCV(svc, parameters, n_jobs=-1, cv=3, verbose=1)
        # Check the training time for the SVC
        print("Hyperparameter tuning...")
        t = time.time()
        self.clf.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        print('Best parameters: {}'.format(self.clf.best_params_))
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.clf.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', self.clf.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    def write(self, path):
        # pickle the whole thing
        dist_pickle = {
            "colorspace": self.colorspace,
            "orient": self.orient,
            "pix_per_cell": self.pix_per_cell,
            "cell_per_block": self.cell_per_block,
            "hog_channel": self.hog_channel,
            "clf": self.clf,
            "scaler": self.scaler
        }
        print('Saving model to {}'.format(path))
        pickle.dump(dist_pickle, open(path, "wb"))


def main():
    # fit vehicle classifier
    colorspace = sys.argv[1]  #  'YCrCb'
    hog_channel = sys.argv[2]  # 'ALL'
    kernel = sys.argv[3] # 'linear' 'rbf'
    clf = CarClassifier(colorspace=colorspace, hog_channel=hog_channel, kernel=kernel)
    rand_state = np.random.randint(0, 100)
    clf.fit(
        images=glob.glob('data/*/*/*.png', recursive=True),
        random_state=rand_state)
    # pickle the model to the fs
    clf.write("clf_pickle_{0}_{1}_{2}.p"
              .format(colorspace, hog_channel, kernel))


if __name__ == "__main__":
    main()
