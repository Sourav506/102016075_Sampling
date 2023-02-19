# 102016075_Sampling

The code begins by importing necessary libraries including pandas, seaborn, numpy, sklearn, xgboost, etc. A dataset is loaded from a file called Creditcard_data.txt using the pandas library, and the first five rows of the dataset are printed out.

The data is then split into two sets: training set and testing set using train_test_split() function from sklearn.model_selection. The training set contains 80% of the data, while the testing set contains the remaining 20%. In order to handle the class imbalance issue, the distribution of classes in both the training and testing sets is printed out. The training set is then oversampled using RandomOverSampler from imblearn.over_sampling, followed by undersampling using RandomUnderSampler from imblearn.under_sampling. The ADASYN technique for oversampling is also applied using ADASYN from imblearn.over_sampling.

This code performs oversampling, undersampling, and data augmentation using the ADASYN and SMOTE techniques. The purpose of these techniques is to balance the class distribution in the dataset. The dataset used here is a credit card fraud dataset.

In each of the oversampling and undersampling techniques used, the training set is split into five equal samples, and each sample is used to train five classifiers. The classifiers used include AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, and XGBClassifier from sklearn.ensemble and xgboost libraries. The classification report is printed out for each classifier after each training session.

Overall, the code trains multiple classifiers using different data sampling techniques to deal with the class imbalance issue in the dataset. By comparing the performance of different classifiers trained on the different sampling techniques, the optimal model for handling the class imbalance problem in the dataset can be determined.
