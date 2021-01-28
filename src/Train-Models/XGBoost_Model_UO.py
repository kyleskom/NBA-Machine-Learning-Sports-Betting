from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

data = pd.read_excel('../../Datasets/Full-Data-Set-UnderOver-2020-21.xlsx')
OU = data['OU-Cover']
data.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover'], axis=1, inplace=True)
data = data.values
data = data.astype(float)

for x in tqdm(range(250)):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test)

    param = {
        'max_depth': 7,
        'eta': 0.1,
        'objective': 'multi:softmax',
        'num_class': 3
    }
    epochs = 200

    model = xgb.train(param, train, epochs)

    predictions = model.predict(test)

    acc = round(accuracy_score(y_test, predictions), 3) * 100
    print(acc)
    model.save_model('../../Models/XGBoost_{}%_UO.json'.format(acc))

#xgb.plot_tree(model)
#plt.show()
#plt.savefig("graph.pdf")
