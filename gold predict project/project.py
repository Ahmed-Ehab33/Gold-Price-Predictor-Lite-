import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



data = pd.read_csv(".\gld_price_data.csv")
x = data.iloc [:,[1,3,4,5]].to_numpy()
y = data.iloc [:,[2]].to_numpy()

gold = LinearRegression()
gold.fit(x,y)
(x[0])
(y[0])


print(gold.predict([[ 5086.66 , 73.56, 21.72, 1.09]]))


plt.xlabel('Gold Price (USD)')
plt.ylabel('Frequency')
plt.hist(data['GLD'], bins=25, color='#FFA07A', edgecolor='black')
plt.xlabel('Gold Price (USD)')
plt.ylabel('Frequency')
plt.title('Distribution of Historical Gold Prices')
plt.grid(True)
plt.show()