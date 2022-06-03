from sklearn import linear_model
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator


df = pd.read_csv("FP.csv")

# take a look at the dataset
#df.head()

#use required features
cdf = df[['Year','Month','day','hours','minutes','Day_of_Week','weekend','electric_power']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :7]
y = cdf.iloc[:, -1]

exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=19, min_samples_split=20)),
    MaxAbsScaler(),
    KNeighborsRegressor(n_neighbors=3, p=1, weights="distance")
)


#Fitting model with trainig data
exported_pipeline.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(exported_pipeline, open('model.pkl', 'wb'))


#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2.6, 8, 10.1]]))

