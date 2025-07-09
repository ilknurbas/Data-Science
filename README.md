
# Data-Science-Projects

This repository presents a collection of data science tasks and mini-projects developed using Python and R, covering topics such as classification, dimensionality reduction, concept drift adaptation, and regression analysis. Each numbered section showcases a standalone project with its own objective, methodology, and results.

**`01: Handwritten Digit Recognition with Dimensionality Reduction`**

This task showcases the implementation of a digit classification pipeline using the MNIST-style dataset of 5,000 handwritten digit samples. The project explores dimensionality reduction techniques—PCA, LDA, t-SNE, and Sammon's Mapping—to visualize high-dimensional data and evaluate their impact on classification accuracy using a Gaussian Naive Bayes model.

**`02: Fall Detection with Wearable Sensor Data `**

This task demonstrates fall classification using wearable sensor signals. It includes both unsupervised and supervised approaches to detect fall versus non-fall actions from a 566×306 dataset. Principal Component Analysis (PCA) is applied for dimensionality reduction and clustering analysis, followed by classifier training with Support Vector Machines (SVM) and Multi-layer Perceptron (MLP). Classification results show consistent high accuracy across methods, highlighting the potential of sensor-based fall detection.

**`03: Data Stream Mining – Concept Drift and Online Classification `**

This task explores online classification techniques and concept drift handling in data stream environments using synthetic SEA datasets. Multiple learning approaches were evaluated, including:
* Online single classifiers: MLP, KNN, Hoeffding Tree
* Ensemble methods: Majority Voting (MV), Weighted Majority Voting (WMV), GOOWE
* Batch classifiers: MLP, KNN, HT (for baseline comparison) 

The system investigates performance under various noise levels and drift conditions, applying evaluation strategies such as prequential accuracy tracking. Concept drift adaptation and diversity measures in ensembles were analyzed based on accuracy outcomes.


**`04: Dodgers Game Attendance – Linear Regression in R `**

This task contains a linear regression model built in R to analyze and predict attendance at Los Angeles Dodgers home games. The model incorporates variables such as game day, weather conditions, promotional events (e.g., bobblehead giveaways), and opposing teams to estimate fan turnout. The project aims to uncover patterns that influence stadium attendance and provide actionable insights for event planning.

