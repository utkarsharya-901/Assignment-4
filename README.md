# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
A healthcare organization wants to develop a machine learning model to predict whether a breast tumor is Malignant (M) or Benign (B) based on diagnostic measurements. The primary objective is to develop a K-Nearest Neighbors (KNN) classification model to classify these tumors accurately.

## Dataset Link
Breast Cancer Wisconsin Diagnostic Dataset (Kaggle): 
[https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
*(Note: Dataset is not uploaded directly to comply with licensing and assignment instructions).*

## Libraries Used
* Pandas (Data manipulation and loading)
* Scikit-Learn (Model building, preprocessing, and evaluation)
* Matplotlib & Seaborn (Visualization of the confusion matrix)

## Methodology
1. **Data Preprocessing:** Handled missing values (dropped completely empty `Unnamed: 32` column), removed the non-predictive `id` column, and encoded the `diagnosis` target variable (`M`=1, `B`=0).
2. **Data Splitting:** Divided the dataset into 80% training data and 20% testing data.
3. **Scaling:** Standardized the feature values using `StandardScaler` to ensure uniform distance calculations.
4. **Model Development:** Trained a K-Nearest Neighbors classifier utilizing an initial neighbor count of K=5.

## Results
* **Accuracy Score:** 0.9474
* **Precision:** 0.9302
* **Recall:** 0.9302
* **F1-Score:** 0.9302

## Conclusion
This project successfully developed a machine learning model to predict whether a breast tumor is Malignant or Benign based on diagnostic measurements. Using the K-Nearest Neighbors (KNN) algorithm initialized at K=5, the model achieved a 94.7% accuracy and a 93% F1-score on the testing set. Feature scaling (standardization) was crucially important in this workflow because KNN relies on calculating Euclidean distances between data points. Without scaling, features with naturally larger numeric ranges would disproportionately dominate the distance calculation over smaller-range features, skewing the predictions. One limitation of the KNN algorithm observed is its nature as a "lazy learner"—it does not learn a discriminative function but rather memorizes the dataset, making the inference computationally heavy and slow as the dataset size grows.
