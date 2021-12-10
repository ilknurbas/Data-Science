# GE461: Introduction to Data Science
# Assignment for Data Stream Mining
# name : İlknur Baş
# id :  21601847

# REFERENCES
# [1] https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.data.SEAGenerator.html
# [2] https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.trees.HoeffdingTreeClassifier.html#skmultiflow.trees.HoeffdingTreeClassifier
# [3] https://scikit-multiflow.readthedocs.io/en/stable/api/generated/skmultiflow.lazy.KNNClassifier.html#skmultiflow.lazy.KNNClassifier
# [4] https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# [5] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.lazy import KNNClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.neural_network import MLPClassifier

print("Assignment for Data Stream Mining")

# Reference [1] is used for the generation of datasets.
print("1. Dataset Generation")
print("a. SEA Dataset")

sea_generate_A = SEAGenerator(classification_function=2)
print("Number of features: ", sea_generate_A.n_num_features)

# Generating a dataset with 20,000 instance
sea_generate_a = sea_generate_A.next_sample(20000)
print("sea_generate_A: ", sea_generate_a)

# sea_dataset includes 2 arrays, one for labels (0/1) other for the data
# The following code is only for a better understanding of the concepts.
data, label = sea_generate_a
print("Data: ", data)
print("Label: ", label)

print("b. SEA Dataset10")
sea_generate_B = SEAGenerator(classification_function=2, noise_percentage=0.1)
sea_generate_b = sea_generate_B.next_sample(20000)

print("c. SEA Dataset70")
sea_generate_C = SEAGenerator(classification_function=2, noise_percentage=0.7)
sea_generate_c = sea_generate_C.next_sample(20000)

print("2. Data Stream Classification with Three Separate Online Single Classifiers: HT, KNN, MLP")

# Reference [2] is used for this part.
print("HT starting...")
hoeffding = HoeffdingTreeClassifier()

# Training is being done with the help of Hoeffding Tree Classifier.
# Logic of the following code is written in reference [2].
datasets = [sea_generate_A, sea_generate_B, sea_generate_C]
for a in datasets:
    count = 0
    # Since there is 20000 instances in one dataset,each of them should be check
    # while finding the accuracy.
    for k in range(0, 20000):
        # array y has the label values such as 0 or 1.
        X, y = a.next_sample()
        prediction = hoeffding.predict(X)
        if y[0] == prediction[0]:
            count += 1
        hoeffding = hoeffding.partial_fit(X, y)

    print("Dataset with noise percentage: ", a.noise_percentage)
    print("Accuracy: ", float(count/20000))

# Reference [3] is used for this part.
print("KNN starting...")
knn = KNNClassifier()

# Training is being done with the help of KNN Classifier.
# Logic of the following code is written in reference [3].
for a in datasets:
    count = 0
    for k in range(0, 20000):
        # array y has the label values such as 0 or 1.
        X, y = a.next_sample()
        prediction = knn.predict(X)
        if y[0] == prediction[0]:
            count += 1
        knn = knn.partial_fit(X, y)
    print("Dataset with noise percentage: ", a.noise_percentage)
    print("Accuracy: ", float(count / 20000))


# Reference [4] is used for this part.
# Actually I am not very sure about this part. Similar code piece is used
# for batch classification. The documentation was not saying much.
print("MLP starting...")
clf = MLPClassifier(hidden_layer_sizes=(200, 4))
datasetOther = [sea_generate_a, sea_generate_b, sea_generate_c]
select = 0
for a in datasetOther:
    train_X, test_X, train_Y, test_Y = train_test_split(a[0], a[1],)
    clf.fit(train_X, train_Y)
    count = 0
    for i in range(0, 5000):
        if clf.predict(test_X)[i] == test_Y[i]:
            count = count + 1
    select = select+1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", (count / float(len(test_Y))))

# Reference [5] is used for this part.
print("3. Data Stream Classification with Two Online Ensemble Classifiers: MV, WMV")
datasetOther = [sea_generate_a, sea_generate_b, sea_generate_c]
print("a: Majority voting rule MV")
# Also by default voting feature's value is hard.
voting = VotingClassifier(estimators=[('HT', hoeffding), ('KNN', knn), ('MLP', clf)], voting='hard')

select = 0
for a in datasetOther:
    voting.fit(a[0], a[1])
    prediction = voting.predict(a[0])
    # Accuracy score function is to find the accuracy as percentage.
    accuracy = accuracy_score(a[1], prediction)

    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ",  accuracy)

print("b: Weighted majority voting rule WMV")
# I determined the value of weights parameter.
votingWMV = VotingClassifier(estimators=[('HT', hoeffding), ('KNN', knn), ('MLP', clf)], weights=[1, 3, 2])
select = 0
for a in datasetOther:
    votingWMV.fit(a[0], a[1])
    prediction = votingWMV.predict(a[0])
    # Accuracy score function is to find the accuracy as percentage.
    accuracy = accuracy_score(a[1], prediction)

    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ",  accuracy)


# Batch Classification is done by splitting and training the data
print("4. Batch Classification with Three Separate Batch Single Classifiers: HT, KNN, MLP ")
datasetOther = [sea_generate_a, sea_generate_b, sea_generate_c]
print("Batch Classification with HT: ")
select = 0
for a in datasetOther:
    # Split the data
    train_1, test_1, train_11, test_11 = train_test_split(a[0], a[1], )
    hoeffding.fit(train_1, train_11)
    prediction = hoeffding.predict(test_1)
    accuracy = accuracy_score(test_11, prediction)
    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", accuracy)

print("Batch Classification with MLP: ")
select = 0
for a in datasetOther:
    # Split the data
    train_1, test_1, train_11, test_11 = train_test_split(a[0], a[1], )
    clf.fit(train_1, train_11)
    prediction = clf.predict(test_1)
    accuracy = accuracy_score(test_11, prediction)
    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", accuracy)

print("Batch Classification with KNN: ")
select = 0
for a in datasetOther:
    # Split the data
    train_1, test_1, train_11, test_11 = train_test_split(a[0], a[1], )
    knn.fit(train_1, train_11)
    prediction = knn.predict(test_1)
    accuracy = accuracy_score(test_11, prediction)
    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", accuracy)

# Batch Classification is done by splitting and training the data
print("5. Batch Classification with Two Batch Ensemble Classifiers: MV, WMV ")
datasetOther = [sea_generate_a, sea_generate_b, sea_generate_c]
print("Batch Classification with WMV: ")
select = 0
for a in datasetOther:
    # Split the data
    train_1, test_1, train_11, test_11 = train_test_split(a[0], a[1], )
    votingWMV.fit(train_1, train_11)
    prediction = votingWMV.predict(test_1)
    accuracy = accuracy_score(test_11, prediction)
    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", accuracy)

print("Batch Classification with MV: ")
select = 0
for a in datasetOther:
    # Split the data
    train_1, test_1, train_11, test_11 = train_test_split(a[0], a[1], )
    voting.fit(train_1, train_11)
    prediction = voting.predict(test_1)
    accuracy = accuracy_score(test_11, prediction)
    select = select + 1
    if select == 1:
        print("Dataset with noise percentage: ", 0)
    elif select == 2:
        print("Dataset with noise percentage: ", 0.1)
    elif select == 3:
        print("Dataset with noise percentage: ", 0.7)
    print("Accuracy: ", accuracy)
