# Cognibot-ML-Project
## Introduction
Covertype Prediction is an actual forest cover type dataset for an observation (30 x 30 meter cell) which was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. This dataset includes information on tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography. There are over half a million measurements in total. This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

Given elevation, aspect, slope, tree type, shadow coverage, distance to nearby landmarks (roads etcetera), soil type, and local topography data  can we predict what type of tree would be present in forest ? Our project mainly focuses on the fact to predict the type of trees in sections of wooden area. Analyzing forest composition is a valuable aspect of maintaining a healthy wilderness around the area. Forest cover type data is often collected by hand or computing technologies, e.g. satellite imaging of a particular area well according to our data there are four wilderness areas located in the Roosevelt National Forest of northern Colorado. In this report, we aim to predict forest cover type using machine learning model, e.g. Random Forest and a variety of classification algorithms.
## Experimental or Materials and Methods, Algorithms Used
Data science provides a plethora of classification algorithms such as logistic regression, support vector machine, naive Bayes classifier, and decision trees. But near the top of the classifier hierarchy is the random forest classifier (there is also the random forest regressor but that we are not using in our project).

### Train Test Split
The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model. The procedure involves taking a dataset and dividing it into two subsets. The first subset is used to fit the model and is referred to as the training dataset. The second subset is not used to train the model; instead, the input element of the dataset is provided to the model, then predictions are made and compared to the expected values. This second dataset is referred to as the test dataset.
 
According to our data there are 581012 rows and 11 columns used in our project including cover type variable. X variable in splitting is dropping cover type variable rest as it is been separated. y variable in splitting is only cover type variable used. If we want to check the splitting is been correctly then we used X.shape (581012, 10) and y.shape (591012, 1). Next, we can split the dataset so that 75 percent is used to train the model and 25 percent is used to evaluate it. This split was chosen arbitrarily.

### Random Forest Algorithm
Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. The low correlation between models is the key. Uncorrelated models can produce ensemble predictions that are more accurate than any of the individual predictions. The reason for this wonderful effect is that the trees protect each other from their individual errors. 

The random forest algorithm establishes the outcome based on the predictions of the decision trees. It predicts by taking the average or mean of the output from various trees. Increasing the number of trees increases the precision of the outcome. A random forest eradicates the 
limitations of a decision tree algorithm. It reduces the overfitting of datasets and increases precision.

Classification in random forests employs an ensemble methodology to attain the outcome. The training data is fed to train various decision trees. Dataset consists of observations and features that will be selected randomly during the splitting of nodes. Random Forest is not ideal in situation like extrapolation (type of estimation, beyond the original observation range, of the value of a variable on the basis of its relationship with another variable) and in the case where data is very sparse (Data which has relatively high percentage of the variable's cells do not contain actual data such as “empty” or NA). 

Random Forest is very useful in places like where it can handle large datasets efficiently, produces good predictions and produces higher accuracy than decision trees. But in some places Random Forest is not that good like where more resources are required for computation and most importantly consumes more time compared to decision tree algorithm.

### Creating and Training Model
As many supervised learning algorithms can be used like Logistic Regression and Support Vector Machines (SVM) as well as some unsupervised learning algorithms like Principal Component Analysis (PCA) and K-Means Clustering. But the best and more efficient algorithm is none other than Random Forest itself. As it can give good predictions with high score which can properly be fit in our model. So, we decided to import RandomForestClassifier from scikit-learn. 

Now we were ready to create our model using RandomForestClassifier with some basic parameters like:- 
1. n_estimators – number of trees in forest.
2. class_weight – weights associated with classes (n_samples / (n_classes * np.bincount(y))) for “balanced” mode.
3. n_jobs - number of jobs to run in parallel.
4. random_state - controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider   when looking for the best split at each node (if max_features < n_features).
5. bootstrap – ensures whether bootstrap samples are used when building trees and if False, then the whole dataset is used to build each tree.

Now it’s time for some training using “.fit()” in our model with X_train and y_train inside fit it fastens our model that’s how we trained our model very efficiently.

## Results and Discussion, Performance Analysis
Data Science and Machine Learning has a very important feature to check our accuracy score and most importantly predictions plays an important role in our model. If classification is about separating data into classes, prediction is about fitting a shape that gets as close to the data as possible.

Let’s talk about Prediction in Machine Learning and how it plays an important step in our project. So, “Prediction” refers to the output of an algorithm after it has been trained on a historical dataset and applied to new data when forecasting the likelihood of a particular outcome. In our project we used predict() function of scikit-learn to predict our model taking X_test inside to predict() function and called it y_predict. For our sake we checked the length of our y_predict which gave us 1,45,253 which is equal to the length of X_test before training and predicting our model. So, as far as we are going in right direction towards the end of our project.

We imported accuracy_score, classification_report and confusion_matrix from scikit-learn metrics to check aur models performance after predicts done on our model. It gave us some very excellent results with accuracy of 95.20%.

The classification report visualizer displays the precision, recall, F1, and support scores for the model.

The confusion matrix is a matrix used to determine the performance of the classification models for a given set of test data. We used heatmap to do our visualization of our confusion matrix of y_test and our y_predict to see better results. We describe predicted values as Positive and Negative and actual values as True and False.

Accuracy, precision and recall should be high as possible. It is difficult to compare two models with low precision and high recall or vice versa. So, to make them comparable, we use F-Score. F-score helps to measure Recall and Precision at the same time.

### Thus, we were pleased to see that our model was not struggling with overfitting and during classification problems to check our model we used some nice techniques to check our model too.

## References
- CoverType Data Set [UCI](https://archive.ics.uci.edu/ml/datasets/covertype)
- Research Papers 
   - [Paper 1](https://cseweb.ucsd.edu/classes/wi15/cse255-a/reports/wi15/Yerlan_Idelbayev.pdf)
   - [Paper 2](http://cs229.stanford.edu/proj2014/Kevin%20Crain,%20Graham%20Davis,%20Classifying%20Forest%20Cover%20Type%20using%20Cartographic%20Features.pdf)
- Kaggle [Link](https://www.kaggle.com/uciml/forest-cover-type-dataset), [Notebook](https://www.kaggle.com/kshitijbarnwal/predictions-using-random-forest), [Example Notebook](https://www.kaggle.com/manisha14/prediction-using-random-forest)
- Scikit-learn [Random Forest](https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Blogs
  - Random Forest
    - [Section](https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/)
    - [TowardsDataScience](https://towardsdatascience.com/understanding-random-forest-58381e0602d2) 
  - [Classification Report](https://medium.com/@kohlishivam5522/understanding-a-classification-report-for-your-machine-learning-model-88815e2ce397)
  - [Confusion Matrix](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62)
