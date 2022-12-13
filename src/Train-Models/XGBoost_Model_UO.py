import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

data = pd.read_excel('../../Datasets/DataSet-2021-22.xlsx')
OU = data['OU-Cover']
data.drop(['Score', 'Home-Team-Win', 'Unnamed: 0', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover'], axis=1,
          inplace=True)
data = data.values
data = data.astype(float)

for x in tqdm(range(100)):
    x_train, x_test, y_train, y_test = train_test_split(data, OU, test_size=.1)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test)

    param = {
        'max_depth': 6,
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 3
    }
    epochs = 300

    model = xgb.train(param, train, epochs)

    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y), 3) * 100
    print(acc)
    model.save_model('../../Models/XGBoost_{}%_UO-6.json'.format(acc))
