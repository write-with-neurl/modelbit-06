import modelbit, sys
from typing import *
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.utils._bunch import Bunch

model = modelbit.load_value("data/model.pkl") # LogisticRegression()
iris = modelbit.load_value("data/iris.pkl") # {'data': array([[5.1, 3.5, 1.4, 0.2], [4.9, 3. , 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5. , 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5. , 3.4, 1.5, 0.2], [4.4, 2....

# main function
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict the species of an iris flower using logistic regression.

    Args:
    sepal_length (float): Sepal length in cm.
    sepal_width (float): Sepal width in cm.
    petal_length (float): Petal length in cm.
    petal_width (float): Petal width in cm.

    Returns:
    str: Predicted species of the iris.
    """
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction][0]

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   print(predict_iris_species(*(modelbit.parseArg(v) for v in sys.argv[1:])))