from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd


bestsellers_data= pd.read_csv("E:/Destinee's/Data/AmazonBestSeller/archive/amazonbestsellers.csv")

print(bestsellers_data.head())
print(bestsellers_data.shape)

#Clean Data
bestsellers_data.drop_duplicates(inplace = True)

bestsellers_data.rename(columns={"Name":"Title","Year":"Publication Year", "User Rating":"Rating"}, inplace = True)
bestsellers_data["Price"]=bestsellers_data["Price"].astype(float)

# Set y 
y = bestsellers_data.Price
#Create List of features for price modeling
book_features = ["Rating","Reviews"]
X = bestsellers_data[book_features]

#test
print(X.describe())
print(y.describe())

#price modeling based on rating and reviews
price_model = DecisionTreeRegressor(random_state=1)
print(price_model.fit(X,y))
price_predictions = price_model.predict(X)
print(price_predictions)
print(mean_absolute_error(y,price_predictions))

#split data to use some as training data and some as validation
train_X,val_X,train_y,val_y = train_test_split(X,y,random_state=0)
price_model.fit(train_X,train_y)
val_predictions = price_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))