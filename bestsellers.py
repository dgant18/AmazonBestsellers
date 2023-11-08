import pandas as pd

df= pd.read_csv("E:\Destinee's\Data\AmazonBestSeller\\archive\\amazonbestsellers.csv")

print(df.head())
print(df.shape)

#Clean Data
df.drop_duplicates(inplace = True)

df.rename(columns={"Name":"Title","Year":"Publication Year", "User Rating":"Rating"}, inplace = True)
df["Price"]=df["Price"].astype(float)