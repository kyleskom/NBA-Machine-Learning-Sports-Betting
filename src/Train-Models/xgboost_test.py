from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_excel('../../Datasets/TEST123.xlsx')
scores = data['Score']
margin = data['Home-Team-Win']
data.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU', 'OU-Cover'], axis=1, inplace=True)
data = data.values
data = data.astype(float)

x_train, x_test, y_train, y_test = train_test_split(data, margin, test_size=.1)

train = xgb.DMatrix(x_train, label=y_train)
test = xgb.DMatrix(x_test, label=y_test)

param = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': 2
}
epochs = 100

model = xgb.train(param, train, epochs)

predictions = model.predict(test)

print(accuracy_score(y_test, predictions))

#xgb.plot_tree(model)
#plt.show()
#plt.savefig("graph.pdf")
