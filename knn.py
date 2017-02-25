# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:33:31 2017

@author: abrown09
"""
#%% packages
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import neighbors
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.cross_validation import train_test_split

#%% data
iris_df = pd.read_table('/Users/amybrown/Thinkful/Unit_4/Lesson_3/curric-data-001-data-sets/iris/iris.data.csv', sep=',', 
                        names=('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'))

#%% plot sepal length by sepal width
plt.scatter(iris_df['sepal_length'], iris_df['sepal_width'])
# figure out how to plot them in different colors
# figure out how to label axes

#%% pick a new point at random from crosstab of sepal length and width

sepal_df = iris_df[['sepal_length', 'sepal_width']]

obs = sepal_df.sample(n=1)
# selects sample line 108. i don't know how to 'save' this result


#### ATTEMPT 2 ####
#%% functions
# 1) given two data points, calculate the euclidean distance between them
def get_distance(data1, data2):
    points = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
    
def get_neighbours(training_set, test_instance, k):
    distances = [_get_tuple_distance(training_instance, test_instance) for training_instance in training_set]
    # index 1 is the calculated distance between training_instance and test_instance
    sorted_distances = sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]

def _get_tuple_distance(training_instance, test_instance):
    return (training_instance, get_distance(test_instance, training_instance[0]))
    
# 3) given an array of nearest neighbours for a test case, tally up their classes to vote on test case class
def get_majority_vote(neighbours):
    # index 1 is the class
    classes = [neighbour[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0] 
 

train, test = train_test_split(iris_df, test_size=0.3)

X_train = train.drop('class', axis=1)
y_train = train['class']

X_test = test.drop('class', axis=1)
y_test = test['class']

predictions = []
k = 5
for x in range(len(X_test)):
    print('Classifying test instance number ' + str(x) + ":"),
    neighbours = get_neighbours(training_set=train, test_instance=test[x][0], k=5)
    majority_vote = get_majority_vote(neighbours)
    predictions.append(majority_vote)
    print('Predicted label=' + str(majority_vote) + ', Actual label=' + str(test[x][1]))
 
#%% ### ATTEMPT THREE ###

X = X_train.as_matrix(columns=['sepal_length', 'sepal_width'])
y = np.array(y_train)

clf = neighbors.KNeighborsClassifier(10, weights='distance')
trained_model = clf.fit(X, y)
trained_model.score(X, y) # model is 90% accurate

X_test = X_test.as_matrix(['sepal_length', 'sepal_width'])
trained_model.predict(X_test)
trained_model.predict_proba(X_test)
 # implemented KNN but I don't know how to interpret the output
 
#%% #### Attempt 4 ####
target_obs = iris_df.iloc[108]

sepal_list = sepal_df.values.tolist()

distances = euclidean_distances(sepal_list, [[6.7, 2.5]])


d  = distances.reshape((150,1))
d2 = pd.DataFrame(data=d, columns=['distance'])



final = pd.concat([iris_df, d2], axis=1)
final = final.sort_values(by='distance')

#testsort = final.sort_values(by=['distance'], ascending=True)

# now, need to subset the top ten points aside from 108

top10 = final[1:11]

# determine the majority class of the subset
# it totally depends on the4 value of k. if you take the top 10, there is a tie. 
# if you say the top 9, it would be iris-versicolor. Top 3? also Iris-versicolor. 


    # order by the labeled points from nearest to farthest

### knn real attempt 1 ###

X = np.array(iris_df.ix[:, 0:4]) 	# end index is exclusive
y = np.array(iris_df['class']) 	# another way of indexing a pandas df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))

# now, can tune the hyperparameter k

### knn real attempt 1 ###