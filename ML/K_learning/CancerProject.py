# Import the required libraries

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#load breast cancer data and display the features each point represents

breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print("Target")
print(breast_cancer_data.target)
print("\n")

print("Target Names")
print(breast_cancer_data.target_names)
print("\n")

print(breast_cancer_data.target[0])
print('The first data point is benign')

training_data,validation_data,training_labels,validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=50)
                                                                                   
# confirm the split by showing the length of the groups
print("Total number of rows in breast_cancer_data: ", len(breast_cancer_data.data))
print("Number of rows in training_data: ", len(training_data))
print("Number of rows in validation_data: ", len(validation_data))

# create a classifier object with neighbors=3
classifier = KNeighborsClassifier(n_neighbors=3)

# fit the object with training data and labels
classifier.fit(training_data, training_labels)

# check the accuracy by calling score() on the classifer and passing it data from validation set
print(classifier.score(validation_data, validation_labels))

accuracy = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracy.append(classifier.score(validation_data, validation_labels))
    
accuracy

max_value = max(accuracy)
max_index = accuracy.index(max_value)

# print(max_, max_index)
print("The highest accuracy " + str(max_value) + " is when k = " + str(max_index))

# scrolling through this is hard so we will plot them to find the best k for our classifier
plt.plot(range(1, 100), accuracy)
plt.xlabel("k values")
plt.ylabel("validation accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()