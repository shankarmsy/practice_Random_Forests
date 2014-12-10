
# coding: utf-8

# ##Classification
# ###Random Forests

# Random Forests is basically an *ensemble learner* built on Decision Trees. 
# 
# Ensemble learning involves the combination of several models to solve a single prediction problem. It works by generating multiple classifiers/models which learn and make predictions independently. Those predictions are then combined into a single (mega) prediction that should be as good or better than the prediction made by any one classifer.
# 
# Random forest is a brand of ensemble learning, as it relies on an ensemble of decision trees. So before we dive into Random Forests, let's first explore Decision Trees and learn how they work.

# ###Decison Trees
# A decision tree is composed of a series of decisions that can be used to classify an observation in a dataset. They encode a series of binary choices in a process that parallels how a person might classify things themselves, but using an information criterion to decide which question is most fruitful at each step. For instance:
# 
# - Does the shoe fit?
#     - Yes: is it black or brown?
#         - Brown: does it work with Jeans?
#         - Black: is it casual or formal?
#     - No: Is it bigger or smaller?
#         - bigger: is there a smaller size available?
#         - smaller: is there a bigger size available?
#         
# Let's begin this exercise by looking at Decision Trees and why we need Random Forests.    

# In[1]:

#Importing necessary packages in Python
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.learning_curve import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
import seaborn
import urllib
from pprint import pprint
np.random.seed(sum(map(ord, "aesthetics")))
seaborn.set_context('notebook')
seaborn.set_style(style='darkgrid')


# In[2]:

#Let's generate a quick sample dataset in scikit-learn.
X,y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_classes=4, n_clusters_per_class=1, 
                          class_sep=1.0, random_state=0)

#Great, the dataset has 4 classes that we'll try to predict. It's got fairly interesting seperation as we can see below. 

#Let's visualize the data with a scatter plot
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.BuGn)
plt.show()


# In[3]:

#Great, let's now fit this dataset to the Decision Tree Classifier and see how well it does.
dtree = DecisionTreeClassifier(max_depth=10).fit(X,y) #this parameter defines the maximum depth of the tree
y_pred=dtree.predict(X)

print metrics.classification_report(y, y_pred)

#THe report tells us that the overall accuracy of the predicted labels is about 94%. Looking at the data, we can be
#almost certain that this is definitely overfitting. To predict 94% of this dataset correctly, the tree would need to be
#extremely well tuned to the dataset we trained on (for now, the entire X dataset). This will mean that when you expose
#new data to the model, it will not be able to predict so well.

#We can confirm our understanding by doing a train/cv split on the data. Let's define a couple of functions next
#that will help us run this multiple times. We'll begin by doing a 80/20 split on the data below.
X_train, X_test, y_train, y_test = train_test_split(X,y)


# In[4]:

#All right let's do this the right way. We'll use a cross-validation generator to select train and CV datasets to finetune
#parameters such as C (Regularization parameter we saw earlier). These hyperparameters are extremely critical to the model.
#Now, if we tune parameters against the Test dataset, we will end up biasing towards the test set and will once again
#not generalize very well. We will also have no good way to find out since we have essentially trained on all our data.

#Luckily scikit-learn has builit-in packages that can help with this. We'll use a crossvalidation generator that
#can train the model by tuning the parameters based on a cross-validation subset (cv) that is picked from within the 
#training set. A different cv subset will be picked for each iteration, we control the number of iterations. Then we will 
#use these cv/train splits and run a gridsearch function that will evaluate the model with each split and tune parameters 
#to give us the best parameter that gives the optimal result.

#Defining this as a function so we can call it anytime we want

def fit_trees(algo, n_jobs, max_depth, n_estimators):

#Choose Estimator as Decision Trees or Random Forests
    if algo == "Decision Trees":
        estimator = DecisionTreeClassifier()
    else:
        estimator = RandomForestClassifier()

#Choose cross-validation generator - let's choose ShuffleSplit which randomly shuffles and selects Train and CV sets
#for each iteration. There are other methods like the KFold split.
    cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2)

#Apply the cross-validation iterator on the Training set using GridSearchCV. This will run the classifier on the 
#different train/cv splits using parameters specified and return the model that has the best results

#Note that we are tuning based on the F1 score 2PR/P+R where P is Precision and R is Recall. This may not always be
#the best score to tune our model on. I will explore this area further in a seperate exercise. For now, we'll use F1.

    if algo == "Decision Trees":
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(max_depth=max_depth),
                                  n_jobs=n_jobs, scoring='f1')
    else:
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(n_estimators=n_estimators, 
                                  max_depth=max_depth), n_jobs=n_jobs, scoring='f1')

#Also note that we're feeding multiple neighbors to the GridSearch to try out.

#We'll now fit the training dataset to this classifier
    classifier.fit(X_train, y_train)

#Let's look at the best estimator that was found by GridSearchCV
    print "Best Estimator learned through GridSearch"
    print classifier.best_estimator_
    
    return cv, classifier.best_estimator_.max_depth, classifier.best_estimator_.n_estimators


# In[5]:

#Below is a plot_learning_curve module that's provided by scikit-learn. It allows us to quickly and easily visualize how
#well the model is performing based on number of samples we're training on. It helps to understand situations such as 
#high variance or bias.

#We'll call this module in the next segment. 

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[20]:

#WARNING - THIS MIGHT TAKE A WHILE TO RUN. TRY ADJUSTING parameters such as n_jobs (jobs to run in parallel, before 
#increasing this make sure your system can handle it), n_iter for ShuffleSplit (in the function definition) and reducing 
#number of values being tried for max_depth/n_estimators.

#SELECT INTERRUPT IN THE MENU AND PRESS INTERRUPT KERNEL IF YOU NEEDD TO STOP EXECUTION

max_depth=np.linspace(5,10,5)

#Let's fit SVM to the digits training dataset by calling the function we just created.
cv,max_depth,n_estimators=fit_trees('Decision Trees', n_jobs=5, max_depth=max_depth, n_estimators=0)

#Great, looks like the grid search returned a best fit with max_depth = 7.5
#Let's look at the learning curve to see if there's overfiting. For more information about learning curves, read the
#function definition above.


# In[22]:

#We'll call the plot_learning_curve module by feeding it the estimator (best estimator returned from GS) and train/cv sets.

#The module simply runs the estimator multiple times on subsets of the data provided and plots the train and cv scores.
#Note that we're feeding the best parameters we've learned from GridSearchCV to the estimator now.
#We may need to adjust the hyperparameters further if there is overfitting (or underfitting, though unlikely)
title = "Learning Curves (Decision Trees, max_depth=%.6f)" %(max_depth)
estimator = DecisionTreeClassifier(max_depth=max_depth)
plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
plt.show()

#There's a pitch-perfect illustration of overfitting. Look at the gulf between the training and cv scores. As we train
#on more and more examples, the training score does decrease and cv scores increases but we'll need exponentially more
#examples to reduce the gulf between the two. Let's confirm understanding by looking at the test scores.


# In[23]:

#Let's see how our trained model performs on the test set. We are not going to train on this set merely looking at how well
#our model can generalize.

#Calling Fit on the estimator object so we can predict. We're NOT retraining the classifier here.
estimator.fit(X_train, y_train)
y_pred=estimator.predict(X_test)
print metrics.classification_report(y_test,y_pred)
print "Decision Trees: Final Generalization Accuracy: %.6f" %metrics.accuracy_score(y_test,y_pred)

#That's not too bad but we can get a much better result if we addressed the overfitting problem. Let's now try the random
#forests classifier to see how it does.


# In[25]:

#WARNING - THIS MIGHT TAKE A WHILE TO RUN. TRY ADJUSTING parameters such as n_jobs (jobs to run in parallel, before 
#increasing this make sure your system can handle it), n_iter for ShuffleSplit (in the function definition) and reducing 
#number of values being tried for max_depth/n_estimators.

#SELECT INTERRUPT IN THE MENU AND PRESS INTERRUPT KERNEL IF YOU NEEDD TO STOP EXECUTION

max_depth=np.linspace(5,10,5)
n_estimators=[10, 100, 1000]

#Let's fit SVM to the digits training dataset by calling the function we just created.
cv,max_depth,n_estimators=fit_trees('Random Forests', n_jobs=10, max_depth=max_depth, n_estimators=n_estimators)

#Great, looks like the grid search returned a best fit with n_estimators=1000
#Let's look at the learning curve to see if there's overfiting. For more information about learning curves, read the
#function definition above.


# In[13]:

#Okay, let's plot our learning curve again.
title = "Learning Curves (Random Forests, n_estimators=%.6f)" %(n_estimators)
estimator = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
plt.show()

#OK that doesn't look very convincing but it's probably harder to see but RF does address overfitting. One thing we can
#observe is that the training set scores don't drop like before while the cv score also looks to have plateued. Other 
#aspects like attribute selection (which features are important?) matter as well and since our dataset is pretty limited,
#we're not able to gauge the full extent of RF's performance. Let's move onto a bigger and more complex dataset
#and find out.


# In[28]:

#Let's see how our trained model performs on the test set. We are not going to train on this set merely looking at how well
#our model can generalize.

#Calling Fit on the estimator object so we can predict. We're NOT retraining the classifier here.
estimator.fit(X_train, y_train)
y_pred=estimator.predict(X_test)
print metrics.classification_report(y_test,y_pred)
print "Random Forests: Final Generalization Accuracy: %.6f" %metrics.accuracy_score(y_test,y_pred)

#Before we move on, let's look at a key parameter that RF returns, namely feature_importances. This tells us which
#features in our dataset seemed to matter the most (although won't matter in the present scenario with only 2 features)
print estimator.feature_importances_

#So both features seem to be almost equally important.


# ###Statlog (German Credit Data) Data Set
# This dataset hosted & provided by the UCI Machine Learning Repository contains mock credit application data of customers. Based on the attributes provided in the dataset, the customers are classified as good or bad and the labels will influence credit approval. The dataset contains several attributes such as:
# - Credit History
# - Status of Bank Accounts
# - Employment History
# 
# and several others as would pertain to a credit application. The dataset also requires the use of a cost matrix as follows:
#   
# |(1=Good, 2=Bad)    |Actuals     	| 1 	   | 2    	| 
# |-----------	    |---	        |---	   |---	    |
# | **Predicted** 	| 1 	        | 0 	   | 1    	|
# |           	    | 2 	        | 5 	   | 0   	|
# 
#             
# 
# It is worse to class a customer as good when they are bad (5), than it is to class a customer as bad when they are good (1).
# 
# NOTE: Random Forests has no way of assigning weights to classes - cost sensitive classification as described above. We'll simply good or bad based on on equal weightage to both classes for this example.
# 
# Also this file has already been converted to a format that can be understood by the ML algorithms. We will not be performing data processing in this exercise and will consume the dataset as-is. I'll cover data preparation in detail in a seperate notebook.

# In[2]:

#OK let's get started. We'll download the data from the UCI website.

#Last column is the labels y
url="http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric"
raw_data = urllib.urlopen(url)
credit=np.genfromtxt(raw_data)
X,y = credit[:,:-1], credit[:,-1:].squeeze()
print X.shape, y.shape

#Great, before we do anything else, let's split the data into train/test.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[6]:

#OK, let's fit this dataset to the RF classifier now.

#WARNING - THIS MIGHT TAKE A WHILE TO RUN. TRY ADJUSTING parameters such as n_jobs (jobs to run in parallel, before 
#increasing this make sure your system can handle it), n_iter for ShuffleSplit (in the function definition) and reducing 
#number of values being tried for max_depth/n_estimators.

#SELECT INTERRUPT IN THE MENU AND PRESS INTERRUPT KERNEL IF YOU NEEDD TO STOP EXECUTION

max_depth=np.linspace(5,10,5)
n_estimators=[10, 100, 500, 1000]

#Let's fit SVM to the digits training dataset by calling the function we just created.
cv,max_depth,n_estimators=fit_trees('Random Forests', n_jobs=10, max_depth=max_depth, n_estimators=n_estimators)

#Great, looks like the grid search returned a best fit with n_estimators=1000 and max_depth=7.5
#Let's look at the learning curve to see if there's overfiting. For more information about learning curves, read the
#function definition above.


# In[7]:

#Okay, let's plot our learning curve.

title = "Learning Curves (Random Forests, n_estimators=%.6f)" %(n_estimators)
estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=10)
plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
plt.show()

#That looks like a reasonable fit with perhaps some overfitting. Let's take a look at the feature_importances 
#to see if we can trim any of the features.


# In[8]:

#Let's call fit on the estimator so we can look at feature importances.
estimator.fit(X_train,y_train)

print "Statlog Credit Data - Feature Importances\n"
print estimator.feature_importances_
print

# Calculate the feature ranking - Top 10
importances = estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in estimator.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Top 10 Features:")

for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
#Plot the feature importances of the forest
indices=indices[:10]
plt.figure()
plt.title("Top 10 Feature importances")
plt.bar(range(10), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(10), indices)
plt.xlim([-1, 10])
plt.show()

#Mean Feature Importance
print "Mean Feature Importance %.6f" %np.mean(importances)


# In[9]:

#Okay so looks like some features are much more important than the others. Attribute 0 for instance is the status of
#checking accounts of the Customer (Importance 13%) while Attribute 7 is how long they've been employed with the 
#current employer (3.6%). It makes sense that one matters more than other.

#Let's do some trimming. We'll try to transform the Training set to include only features that are atleast as important as
#the mean of importances. Let's find out if this improves the accuracy.

#Luckily this is very easy to do in sklearn. RF has a transform method that helps with this.
X_train_r = estimator.transform(X_train, threshold='mean')
X_test_r = estimator.transform(X_test, threshold='mean')

#Let's run the learning curve again.
title = "Learning Curves -Iter2 (Random Forests, n_estimators=%.6f, feature_importances > mean)" %(n_estimators)
estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=10)
plot_learning_curve(estimator, title, X_train_r, y_train, cv=cv)
plt.show()

#Looks like that didn't really have much of an impact on the model. Let's find out how well the model will generalize
#by predicting on the Test dataset.


# In[11]:

#Let's call fit on the estimator so we can look at feature importances.
estimator.fit(X_train_r,y_train)

#Running predictions on Test
y_pred=estimator.predict(X_test_r)
print metrics.classification_report(y_test,y_pred)
print
print "Random Forests: Final Generalization Accuracy: %.6f" %metrics.accuracy_score(y_test,y_pred)


# OK that's a fairly reasonable result. We ran a straightforward Random Forest classifier by cross-validating with a Grid Search then further trimmed the features by looking at feature_importances and were able to get approximately **80%** accuracy on the Test set.
