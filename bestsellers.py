from sklearn.tree import DecisionTreeRegressor
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
#Create List of features
book_features = ["Rating","Reviews"]
X = bestsellers_data[book_features]

#test
print(X.describe())

#price modeling
price_model = DecisionTreeRegressor(random_state=1)
print(price_model.fit(X,y))
price_predictions = price_model.predict(X)
print(price_predictions)