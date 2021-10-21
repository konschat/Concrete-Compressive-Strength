import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import std, mean
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Define mean absolute percentage erro function as it is not supported
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

data = pd.read_csv('Concrete_Data.csv', sep=",") # load the data

# Define target names
target_names = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregare", "Age", "CC_Strength"]

# Insert target names as columns names
curr_col_names = list(data.columns)

mapper = {}
for i, name in enumerate(curr_col_names):
    mapper[name] = target_names[i]

data = data.rename(columns=mapper)
print(data.head())

# Data Analysis
# Check if there are null values in the data
print(data.isna().sum())

# # Create a grid of Axes n x n, where n : all numeric variables
# sns.pairplot(data)
# plt.show()
#
# corr = data.corr()
# # Correlation between independant variables seems to be limited
# # We check it by plotting the Pearson Correlation coefficients between the features
# sns.heatmap(corr, annot=True, cmap='Blues')
# b, t = plt.ylim()
# plt.ylim(b+0.5, t-0.5)
# plt.title("Feature Correlation Heatmap")
# plt.show()
#
# ax = sns.distplot(data.CC_Strength)     # Moving with the EDA(Exploratory Data Analysis)
# ax.set_title("Compressive Strength Distribution") #Also ending data exploration

# Data Splitting/Models/Trainings/Predictions
X = data.iloc[:,:-1]         # Features - All columns but last
y = data.iloc[:,-1]          # Target - Last Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
# Standardizing the data i.e. to rescale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)         # The features to have a mean of zero and standard deviation of 1.
X_test = sc.transform(X_test)
print(X.shape, y.shape)

# Training data and making predictions on Test data.
print("[INFO] Building, training, testing the model")

# Define Linear Regression model
lr = LinearRegression()

# Fitting models on Training data / Making predictions on Test data
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_train_predictions = lr.predict(X_train)

                            ################################
                            ########## A) RESULTS ##########
                            ################################

# We proceed to the networks evaluation
print("[INFO] evaluating networks training/testing process...")
print("Model\t\t\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t MAPE")
print("""Testing LinearRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),mean_squared_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_lr), mean_absolute_percentage_error(y_test, y_pred_lr)))
print("""Training LinearRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f} """.format(
        np.sqrt(mean_squared_error(y_train, lr_train_predictions)), mean_squared_error(y_train, lr_train_predictions),
        mean_absolute_error(y_train, lr_train_predictions), mean_absolute_percentage_error(y_train, lr_train_predictions)))

alphas = [0.01, 0.1, 1, 5, 10, 100, 500, 1000]
# Loop to compute the different values of cross-validation scores
for i in range(0, 9):
    if i == 1:
        print("[INFO] Ridge : Alpha regularization and respective scores")
        print("Model\t\t\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t MAPE \t Alpha")
    RidgeModel = Ridge(alpha = i*0.25)
    RidgeModel.fit(X_train, y_train)
    y_pred_ridge = RidgeModel.predict(X_test)
    ridge_train_predictions = RidgeModel.predict(X_train)
    print('Ridge R2: %.3f ' % RidgeModel.score(X_test, y_test))
    print("""Testing RidgeRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
        np.sqrt(mean_squared_error(y_test, y_pred_ridge)), mean_squared_error(y_test, y_pred_ridge),
        mean_absolute_error(y_test, y_pred_ridge), mean_absolute_percentage_error(y_test, y_pred_ridge), alphas[i-1]))
    print("""Training RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
        np.sqrt(mean_squared_error(y_train, ridge_train_predictions)), mean_squared_error(y_train, ridge_train_predictions),
        mean_absolute_error(y_train, ridge_train_predictions), mean_absolute_percentage_error(y_train, ridge_train_predictions),
        alphas[i-1]))


# Loop to compute the different values of cross-validation scores
for i in range(1, 9):
    if i == 1:
        print("[INFO] Lasso : Alpha regularization and respective scores")
        print("Model\t\t\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t MAPE \t Alpha")
    LassoModel = Lasso(alpha = i * 0.25)
    LassoModel.fit(X_train, y_train)
    y_pred_lasso = LassoModel.predict(X_test)
    lasso_train_predictions = LassoModel.predict(X_train)
    print('Lasso R2: %.3f ' % LassoModel.score(X_test, y_test))
    print("""Testing LassoRegression \t {:.2f} \t\t {:.2f} \t\t{:.2f} \t\t{:.2f}\t\t{:.2f}""".format(
        np.sqrt(mean_squared_error(y_test, y_pred_lasso)), mean_squared_error(y_test, y_pred_lasso),
        mean_absolute_error(y_test, y_pred_lasso), mean_absolute_percentage_error(y_test, y_pred_lasso), alphas[i-1]))
    print("""Training LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_train, lasso_train_predictions)), mean_squared_error(y_train, lasso_train_predictions),
            mean_absolute_error(y_train, lasso_train_predictions), mean_absolute_percentage_error(y_train, lasso_train_predictions),
            alphas[i-1]))

# Performance(based on accuracy) report
print('Linear Regression R2: %.3f ' % lr.score(X_test, y_test))     #mean(y_pred_lr), std(y_pred_lr)))
print('Lasso R2: %.3f ' % LassoModel.score(X_test, y_test))
print('Ridge R2: %.3f ' % RidgeModel.score(X_test, y_test))

                            ################################
                            ########## C) RESULTS ##########
                            ################################

# Define those three functions to helps me grasp mse,mape and mae for the validation process.
def calc_train_error(X_train, y_train, model):
    #returns in-sample error for already fit model
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse

def calc_validation_error(X_test, y_test, model):
    #returns out-of-sample error for already fit model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def calc_metrics(X_train, y_train, X_test, y_test, model):
    #fits model and returns the RMSE for in-sample error and out-of-sample error
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    val_mape = mean_absolute_percentage_error(y_test, y_pred)
    val_mae = mean_absolute_error(y_test, y_pred)
    return train_error, validation_error , val_mape, val_mae

# Define k fold cross validation shuffled spliting parameters
sss = ShuffleSplit(n_splits=10, train_size=0.7, test_size=0.3, random_state=42)
sss.get_n_splits(X, y)

for alpha in alphas:
    # Define corresponding tables for the results
    lr_train_errors = []
    lr_validation_errors = []
    lr_val_mapes = []
    lr_val_maes = []

    lr_train_maes = []
    lr_train_mapes = []

    lasso_train_errors = []
    lasso_validation_errors = []
    ridge_train_errors = []
    ridge_validation_errors = []

    lasso_train_mapes = []
    ridge_train_mapes = []
    lasso_val_mapes = []
    ridge_val_mapes = []

    lasso_train_maes = []
    ridge_train_maes = []
    lasso_val_maes = []
    ridge_val_maes = []


    # Split data
    for train_index, val_index in sss.split(X, y):
        X_train2, X_val2 = X.iloc[train_index], X.iloc[val_index]
        y_train2, y_val2 = y.iloc[train_index], y.iloc[val_index]

        # Scale the newly created datasets
        X_train2 = sc.fit_transform(X_train2)
        X_val2 = sc.transform(X_val2)

        # calculate linear errors
        lr_train_error, lr_val_error, lr_mal_mape, lr_val_mae = calc_metrics(X_train2, y_train2, X_val2, y_val2, lr)

        # append to appropriate list
        lr_train_errors.append(lr_train_error)
        lr_validation_errors.append(lr_val_error)
        lr_val_mapes.append(lr_mal_mape)
        lr_val_maes.append(lr_val_mae)

        # Linear Regression MAPE
        lr_predictions = lr.predict(X_train2)
        train_lr_mape = mean_absolute_percentage_error(y_train2, lr_predictions)
        lr_train_maess = mean_absolute_error(y_train2, lr_predictions)

        lr_train_maes.append(lr_train_maess)
        lr_train_mapes.append(train_lr_mape)

        # instantiate model
        lasso = Lasso(alpha=alpha, fit_intercept=True, random_state=42)

        # calculate lasso errors
        lasso_train_error, lasso_val_error, lasso_val_mape, lasso_val_mae = calc_metrics(X_train2, y_train2, X_val2, y_val2, lasso)

        # append to appropriate list
        lasso_train_errors.append(lasso_train_error)
        lasso_validation_errors.append(lasso_val_error)
        lasso_val_mapes.append(lasso_val_mape)
        lasso_val_maes.append(lasso_val_mae)

        # instantiate model
        ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=42)

        # calculate ridge errors
        ridge_train_error, ridge_val_error, ridge_val_mape, ridge_val_mae = calc_metrics(X_train2, y_train2, X_val2, y_val2, ridge)

        # append to appropriate list
        ridge_train_errors.append(ridge_train_error)
        ridge_validation_errors.append(ridge_val_error)
        ridge_val_mapes.append(ridge_val_mape)
        ridge_val_maes.append(ridge_val_mae)

        # Lasso MAPE
        lasso_predictions = lasso.predict(X_train2)
        train_lasso_mape = mean_absolute_percentage_error(y_train2, lasso_predictions)
        lasso_train_maess = mean_absolute_error(y_train2, lasso_predictions)

        lasso_train_maes.append(lasso_train_maess)
        lasso_train_mapes.append(train_lasso_mape)

        # Ridge MAPE
        ridge_predictions = ridge.predict(X_train2)
        train_ridge_mape = mean_absolute_percentage_error(y_train2, ridge_predictions)
        ridge_train_maess = mean_absolute_error(y_train2, ridge_predictions)

        ridge_train_maes.append(ridge_train_maess)
        ridge_train_mapes.append(train_ridge_mape)

        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        lr_scores = cross_val_score(lr, X_val2, y_val2, cv=cv, n_jobs=-1)
        lasso_scores = cross_val_score(lasso, X_val2, y_val2, cv=cv, n_jobs=-1)
        ridge_scores = cross_val_score(ridge, X_val2, y_val2, cv=cv, n_jobs=-1)

    print('Linear Regression Results : MSE (train_error): {:7} (+/-) {}  MSE (val_error): {} (+/-) {} | MAPE (train_error): {:7} (+/-) {}  MAPE (val_error): {} (+/-) {} | MAE (train_error): {:7} (+/-) {} MAE (val_error): {} (+/-) {}'.  # | MAPE (val_error): {}'. # | MAPE (train_error): {:7}' | MAPE (val_error): {}
        format(round(np.mean(lr_train_errors), 2), round(std(lr_train_errors), 2), round(np.mean(lr_validation_errors), 2), round(std(lr_validation_errors), 2),
               round(np.mean(lr_train_mapes), 2), round(std(lr_train_mapes), 2), round(np.mean(lr_val_mapes), 2), round(std(lr_val_mapes), 2), round(np.mean(lr_train_maes), 2),
               round(std(lr_train_maes), 2), round(np.mean(lr_val_maes), 2), round(std(lr_val_maes), 2)))  # , round(np.mean(lasso_train_mapes, 2)) , round(np.mean(lasso_val_mapes), 2)))

    # generate Lasso report
    print('Lasso Results : alpha: {:6} | MSE (train_error): {:7} (+/-) {}  MSE (val_error): {} (+/-) {} | MAPE (train_error): {:7} (+/-) {}  MAPE (val_error): {} (+/-) {} | MAE (train_error): {:7} (+/-) {} MAE (val_error): {} (+/-) {}'. # | MAPE (val_error): {}'. # | MAPE (train_error): {:7}' | MAPE (val_error): {}
          format(alpha, round(np.mean(lasso_train_errors), 2), round(std(lasso_train_errors),2), round(np.mean(lasso_validation_errors), 2), round(std(lasso_validation_errors),2), round(np.mean(lasso_train_mapes),2) , round(std(lasso_train_mapes),2),
                 round(np.mean(lasso_val_mapes), 2), round(std(lasso_val_mapes),2), round(np.mean(lasso_train_maes), 2), round(std(lasso_train_maes), 2), round(np.mean(lasso_val_maes),2), round(std(lasso_val_maes),2) )) # , round(np.mean(lasso_train_mapes, 2)) , round(np.mean(lasso_val_mapes), 2)))

    # generate Ridge report
    print('Ridge Results : alpha: {:6} | MSE (train_error): {:7} (+/-) {}  MSE (val_error): {} (+/-) {} | MAPE (train_error): {:7} (+/-) {}  MAPE (val_error): {} (+/-) {} | MAE (train_error): {:7} (+/-) {} MAE (train_error): {} (+/-) {} '. # | MAPE (train_error): {:7}' | MAPE (val_error): {}
          format(alpha, round(np.mean(ridge_train_errors), 2), round(std(ridge_train_errors),2), round(np.mean(ridge_validation_errors), 2), round(std(ridge_validation_errors),2), round(np.mean(ridge_train_mapes), 2), round(std(ridge_train_mapes),2), round(np.mean(ridge_val_mapes), 2), round(std(ridge_val_mapes),2), round(np.mean(ridge_train_maes), 2), round(std(ridge_train_maes),2), round(np.mean(ridge_val_maes), 2) , round(std(ridge_val_maes),2))) #, round(np.mean(ridge_val_mapes), 2)))       # , round(np.mean(ridge_train_mapes, 2)) , round(np.mean(ridge_val_mapes), 2)


    # Performance report
    print("[INFO] K-Fold validation scores(alpha=default for each regressor)")
    print('Linear Regression Accuracy: %.3f (+/- %.3f)' % (mean(lr_scores), std(lr_scores)))
    print('Lasso Accuracy: %.3f (+/- %.3f)' % (mean(lasso_scores), std(lasso_scores)))
    print('Ridge Accuracy: %.3f (+/- %.3f)' % (mean(ridge_scores), std(ridge_scores)))



                            ################################
                            ########## D) RESULTS ##########
                            ################################

def test_poly_regression(X_train, y_train, X_test, y_test, n=2):
    xxx = []
    for i in range(1,n+1):
        poly_reg = PolynomialFeatures(degree=i)
        X_poly = poly_reg.fit_transform(X_train)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y_train)

        if i == 1 :
            # Predicting the 1st result with Polymonial Regression
            y_pol_pred = pol_reg.predict(poly_reg.fit_transform(X_test))  # poly_reg.fit_transform(X_test)

            # Calculating 1st training process's errors
            # returns in-sample error for already fit model
            predictions = pol_reg.predict(X_poly)
            mse = mean_squared_error(y_train, predictions)
            rmse = np.sqrt(mse)

            val_mape = mean_absolute_percentage_error(y_test, y_pol_pred)
            val_mae = mean_absolute_error(y_test, y_pol_pred)

            XX = np.vstack((X_poly, poly_reg.fit_transform(X_test)))
            xxx.append(XX)

            print("[INFO] Polynomial Scores")
            print("Model\t\t\t\t\t\t\t RMSE \t\t MSE \t\t MAE \t\t MAPE")
        else:
            # Predicting a new result with Polymonial Regression above the 1st degree
            y_pol_pred = pol_reg.predict(poly_reg.fit_transform(X_test))

            # Calculating 2nd training process's errors
            # returns in-sample error for already fit model
            predictions = pol_reg.predict(X_poly)
            mse = mean_squared_error(y_train, predictions)
            rmse = np.sqrt(mse)

            val_mape = mean_absolute_percentage_error(y_test, y_pol_pred)
            val_mae = mean_absolute_error(y_test, y_pol_pred)

            XX = np.vstack((X_poly, poly_reg.fit_transform(X_test)))
            xxx.append(XX)

        print("Polynomial Degree : ", i)
        print("""Training PolynomialRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            rmse, mse, val_mae, val_mape))
        print("""Testing PolynomialRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pol_pred)), mean_squared_error(y_test, y_pol_pred),
            mean_absolute_error(y_test, y_pol_pred), mean_absolute_percentage_error(y_test, y_pol_pred)))
        print('Polynomial Accuracy: %.3f (+/- %.3f)' % (mean(y_pol_pred), std(y_pol_pred)))
        if i == n:
            print(XX.shape)
            print(len(xxx))

            with open('output.txt', 'w') as f:
                f.write(str(xxx))

# Call function to operate as expected
test_poly_regression(X_train, y_train, X_test, y_test, 10)
