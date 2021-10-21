# Concrete compressive strength

In this problem we apply the linear regression method to the problem of modeling the compressive strength of concrete. The concrete (cement) is the
most important material in building construction, and its compressive strength is a nonlinear function of its age and its components. In this problem, we try to train a model of concrete compressive strength as a function of its characteristics, given the dataset "Concrete_Data.csv". Also given is a file "Concrete_Readme.txt" with information about the dataset and a file "Concrete_Data.xls" which has the name of each characteristic (it has been removed from Concrete_Data.csv for convenience).
The variable we are trying to model corresponds to the last column, and is "Concrete compressive strength", which corresponds to the compressive strength of the concrete and is given in MPa. Initially we will use 70% of the dataset for training (in the order given) and the remaining 30% for evaluation (testing).
The following are addressed:
Α. Evaluate the performance of Ordinary Least Squares regression, as well as Ridge and LASSO linear regression. Experiment with different values of the normalisation weight. Summarize the results of the evaluation based on mean squared error (MSE), mean absolute error (MAE) and mean absolute percentage error (MAPE).

                                                                              𝑀𝑆𝐸=1𝑁||𝒚-𝒚̂||22

                                                                              𝑀𝐴𝐸=1𝑁||𝒚-𝒚̂||1

                                                                         𝑀𝐴𝑃𝐸=1𝑁Σ|𝑦(𝑖)-𝑦̂(𝑖)𝑦(𝑖)𝑁𝑖=1|
                                   
B. In the previous question we chose the normalization weight (hyperparameter alpha in scikit-learn) by looking at the results in the evaluation set(underlying disadvantages).

C. Repeat step A, except that the evaluation will be done as follows: Randomly select 70% of the data set for training and the remaining 30% for evaluation and calculate the evaluation metrics. This process is repeated 10 times and as a result we provide the mean and standard deviation of each metric(compare it with step B). 

D. Given the nonlinearity of the function we are trying to model, it is worth evaluating more expressive linear regression models with polynomial terms of the attributes. Implement function test_poly_regression(X_train, y_train, X_test, y_test, n=2) which will take as input a training set (X_train design matrix of the training set and y_train the dependent variable), an evaluation set (X_test, y_test), and a polynomial degree 𝑛≥1. The function should generate a new feature set consisting of the original features and versions of them raised to powers up to 𝑛. Specifically, if 𝐗 is the original set, the function creates the set 𝚾𝑛=[𝐗 𝐗𝟐 ... 𝐗𝐧]
This is the case for both the training set and the evaluation set. Then, the function trains and evaluates linear regression models on the resulting sets. Perform this procedure for 𝑛=1 to 𝑛=10. You can use any of the evaluation modes from the previous subquestions (fixed training set or random selection iteratively).

Questions based on analysis. 

i) Based on these results, which model would you choose for practical application? 

ii) What are the advantages and disadvantages of models with higher polynomial degrees, 𝑛?
