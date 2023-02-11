import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("/Users/atatekeli/PycharmProjects/AnalysisProjects/vgsales.csv")
print(data.head())

print(data.isnull().sum())

data = data.dropna()

import matplotlib as mpl

game = data.groupby("Genre")["Global_Sales"].count().head(20)
custom_colors = mpl.colors.Normalize(vmin=min(game), vmax=max(game))
colors = [mpl.cm.PuBu(custom_colors(i)) for i in game]
plt.figure(figsize=(7,7))
plt.pie(game, labels=game.index, colors=colors)
central_circle = plt.Circle((0,0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Top 20 Categories of Games Sold", fontsize=20)
plt.show()

print(data.corr())
sb.heatmap(data.corr(), cmap="winter_r")
plt.show()

x = data[["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
y = data["Global_Sales"]

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
print(predictions)
