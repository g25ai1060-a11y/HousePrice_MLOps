from sklearn.tree import DecisionTreeRegressor

from misc import *

df = load_data()

X_train, X_test, y_train, y_test = split_data(df)

model = DecisionTreeRegressor()

model = train_model(
    model,
    X_train,
    y_train
)

mse = evaluate(
    model,
    X_test,
    y_test
)

print(
    "DecisionTree Average MSE:",
    mse
)