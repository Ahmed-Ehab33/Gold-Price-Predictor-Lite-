import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv(".\gld_price_data.csv")
x = data.iloc[ :, [1,3,4,5]].to_numpy()
y = data.iloc[ :, [2]].to_numpy()

model = LinearRegression()
model.fit(x,y)

spx = float(input("Enter SPX value: "))
uso = float(input("Enter USO value: "))
slv = float(input("Enter SLV value: "))
eur_usd = float(input("Enter EUR/USD value: "))

input_data = np.array([[spx,uso,slv,eur_usd]])
prediction = model.predict(input_data)

print( "Predicted Gold price: ",prediction)


y_pred = model.predict(x)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Gold Price')
plt.ylabel('Predicted Gold Price')
plt.title('Actual vs Predicted Gold Prices')
plt.legend()
plt.grid(True)
plt.show()




#done by Ahmed Ehab 
