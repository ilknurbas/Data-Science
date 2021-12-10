import scipy.io
import numpy
from sklearn.model_selection import train_test_split
import sklearn.decomposition
import matplotlib.pyplot
import sklearn.discriminant_analysis
import sklearn.naive_bayes


def print_hi(name):
    print(f'THIS IS PROJECT {name}')


if __name__ == '__main__':
    print_hi('2')

# Read .mat file
# digits:5000x400 labels:5000x1
mat = scipy.io.loadmat('digits.mat')
print("Dataset given-digits:\n", mat['digits'])
print("Dataset given-labels\n", mat['labels'])
array_data = numpy.append(mat['digits'], mat['labels'], axis=1)
print("Dataset given as a numpy array: \n", array_data)

# While searching scikit-learn library, I have encountered with "train_test_split" function.
# The information about it can be found from the link below.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Since in the assignment, it is asked to divide the dataset by selecting half of the patterns,
# I assigned the "test_size" as 0.5. Also I enabled shuffle as true since the selection should be randomly.
train_digits, test_digits, train_labels, test_labels = train_test_split(mat['digits'], mat['labels'], test_size=0.5,
                                                                        shuffle=True)
# In the assignment it is asked to project the 400-dimensional data
# Number of components is 400
# PCA means finding a projection that best represents the data in a least-squares sense.
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
my_pca = sklearn.decomposition.PCA(n_components=400)
# Fitting the model with train_digits.
my_pca.fit(train_digits)

# We are able to get the eigenvalues through the pca.explained_variance_ attribute
# In  the documentation, it is also mentioned that explained_variance_
# is equal to n_components largest eigenvalues of the covariance matrix of X.
# The link is as follows. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
eigenvalues = my_pca.explained_variance_
# Plot the eigenvalues by importing matplotlib.pyplot (https://matplotlib.org/stable/tutorials/introductory/pyplot.html)
matplotlib.pyplot.plot(eigenvalues)
matplotlib.pyplot.show()

# mean_ --> Per-feature empirical mean, estimated from the training set.
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# Sample mean for digits training data
mean_pca = my_pca.mean_
# since it is array with size 400, reshape with 20,20
matplotlib.pyplot.imshow(mean_pca.reshape(20, 20))
matplotlib.pyplot.show()

# In order to display 50 bases (eigenvectors) as images, I did principal component analysis
# for 50 components. (50 is determined by me as I explained in the report.)
pca_for_50_bases = sklearn.decomposition.PCA(n_components=50)
pca_for_50_bases.fit(train_digits)

# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
fig, axs = matplotlib.pyplot.subplots(5, 10)
fig.suptitle('50 bases (eigenvectors) as an image')
myint = 0
for i in range(5):
    for j in range(10):
        axs[i][j].imshow((pca_for_50_bases.components_[myint]).reshape(20, 20))
        myint = myint + 1
        print(f'my int: {myint}')
matplotlib.pyplot.show()

# I choose 20 different subspace dimensions as it is written in the assignment.
# And project both the training data and the test data
pca_array_for_test = [0 for i in range(20)]
pca_array_for_train = [0 for i in range(20)]
pca_array_for_error_test = [0 for i in range(20)]
pca_array_for_error_train = [0 for i in range(20)]
# Subspace_dimension is between 1 and 200
pca_subspace_dimensions = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

for i in range(20):
    print(f'i: {i}')
    my_pca2 = sklearn.decomposition.PCA(n_components=pca_subspace_dimensions[i])
    print(f'pca_subspace_dimensions[i]: {pca_subspace_dimensions[i]}')
    # convert to 1D
    train_labels_1D = train_labels.ravel()
    # Fitting the model with train_digits.
    # Projecting both training and test data
    # test_digits and train_digits was divided by 2 randomly at the beginning of the code
    pca_for_test_data = my_pca2.fit(train_digits).transform(X=test_digits)
    pca_for_train_data = my_pca2.fit(train_digits).transform(X=train_digits)

    # Trained a Gaussian classifier
    # The information about the Gaussian is taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    gaussian = sklearn.naive_bayes.GaussianNB()

    # In order to examine classification error for both the training set and the test set
    # predict(X) method should be called --> Perform classification.

    # convert to 1D
    train_labels_1D = train_labels.ravel()
    gaussian.fit(pca_for_train_data, train_labels_1D)
    pca_classification_test = gaussian.predict(pca_for_test_data)
    pca_classification_train = gaussian.predict(pca_for_train_data)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    pca_accuracy_test = sklearn.metrics.accuracy_score(test_labels, pca_classification_test, normalize=False) / float(
        test_labels.size)
    pca_accuracy_train = sklearn.metrics.accuracy_score(train_labels_1D, pca_classification_train,
                                                        normalize=False) / float(train_labels_1D.size)

    pca_array_for_test[i] = pca_accuracy_test
    pca_array_for_error_test[i] = 1 - pca_accuracy_test
    pca_array_for_train[i] = pca_accuracy_train
    pca_array_for_error_train[i] = 1 - pca_accuracy_train
    print(f'pca_accuracy_test: {pca_accuracy_test}')
    print(f'pca_accuracy_train: {pca_accuracy_train}')

    print(f'pca_error_test: {pca_array_for_error_test[i]}')
    print(f'pca_error_train: {pca_array_for_error_train[i]}')

matplotlib.pyplot.plot(pca_subspace_dimensions, pca_array_for_test, label='PCA-Test-Accuracy Classification Score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
matplotlib.pyplot.plot(pca_subspace_dimensions, pca_array_for_train, label='PCA-Train-Accuracy Classification Score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

matplotlib.pyplot.plot(pca_subspace_dimensions, pca_array_for_error_test, label='PCA-Test-Error Classification Score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
matplotlib.pyplot.plot(pca_subspace_dimensions, pca_array_for_error_train, label='PCA-Train-Error Classification Score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

#### LDA ####

# Performed linear discriminant analysis and used the training data set
# (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html )
my_lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
# convert to 1D
train_labels_1D = train_labels.ravel()
# fit to data
my_lda.fit(X=train_digits, y=train_labels_1D)

# the number of bases that are greater than 10 is giving out-of bounds error, since there are 10 digits
fig2, axs2 = matplotlib.pyplot.subplots(2, 5)
fig2.suptitle('Set of bases')
myint2 = 0
for i in range(2):
    for j in range(5):
        # coef_ is weight vectors
        axs2[i][j].imshow((my_lda.coef_[myint2]).reshape(20, 20))
        myint2 = myint2 + 1
        print(f'my int2: {myint2}')
matplotlib.pyplot.show()

# for each project the data
array_for_test = [0 for i in range(9)]
array_for_train = [0 for i in range(9)]
array_for_error_test = [0 for i in range(9)]
array_for_error_train = [0 for i in range(9)]

# Subspace dimensions between 1 and 9
subspace_dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(9):
    print(f'i: {i}')
    my_lda2 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=subspace_dimensions[i])
    # convert to 1D
    train_labels_1D = train_labels.ravel()
    # projecting both training and test data
    # test_digits and train_digits was divided by 2 randomly at the beginning of the code
    lda_for_test_data = my_lda2.fit(X=train_digits, y=train_labels_1D).transform(X=test_digits)
    lda_for_train_data = my_lda2.fit(X=train_digits, y=train_labels_1D).transform(X=train_digits)

    # Train a Gaussian classifier
    # The information about the Gaussian classifier is taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    gaussian = sklearn.naive_bayes.GaussianNB()

    # in order to examine classification error for both the training set and the test set
    # predict(X) method should be called --> Perform classification on an array of test vectors X.

    # convert to 1D
    train_labels_1D = train_labels.ravel()
    gaussian.fit(lda_for_train_data, train_labels_1D)
    classification_test = gaussian.predict(lda_for_test_data)
    classification_train = gaussian.predict(lda_for_train_data)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    accuracy_test = sklearn.metrics.accuracy_score(test_labels, classification_test, normalize=False) / float(
        test_labels.size)
    accuracy_train = sklearn.metrics.accuracy_score(train_labels_1D, classification_train, normalize=False) / float(
        train_labels_1D.size)

    array_for_test[i] = accuracy_test
    array_for_error_test[i] = 1 - accuracy_test
    array_for_train[i] = accuracy_train
    array_for_error_train[i] = 1 - accuracy_train
    print(f'accuracy_test: {accuracy_test}')
    print(f'accuracy_train: {accuracy_train}')

    print(f'error_test: {array_for_error_test[i]}')
    print(f'error_train: {array_for_error_train[i]}')

matplotlib.pyplot.plot(subspace_dimensions, array_for_test, label=' LDA Test Accuracy classification score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
matplotlib.pyplot.plot(subspace_dimensions, array_for_train, label='LDA Train Accuracy classification score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

matplotlib.pyplot.plot(subspace_dimensions, array_for_error_test, label='LDA Test Error classification score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
matplotlib.pyplot.plot(subspace_dimensions, array_for_error_train, label='LDA Train Error classification score')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()
