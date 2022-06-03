import pandas as pd
from sklearn import linear_model
import pickle
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("FP.csv")

# take a look at the dataset
#df.head()

#use required features
cdf = df[['Year','Month','day','hours','minutes','Day_of_Week','weekend','electric_power']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :7]
y = cdf.iloc[:, -1]

reg=linear_model.LinearRegression()

#Fitting model with trainig data
reg.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(reg, open('model.pkl', 'wb'))


#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2.6, 8, 10.1]]))

