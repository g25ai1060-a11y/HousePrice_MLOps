from sklearn.kernel_ridge import KernelRidge

from misc import *

df = load_data()

X_train, X_test, y_train, y_test = split_data(df)

model = KernelRidge()

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
    "KernelRidge Average MSE:",
    mse
)