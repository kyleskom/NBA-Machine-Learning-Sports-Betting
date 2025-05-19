import NN_Model_ML
current_time = NN_Model_ML.current_time
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{current_time}\n')

import XGBoost_Model_ML
name = XGBoost_Model_ML.name
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{name}\n')

import Logistic_Regression_ML
name_lr = Logistic_Regression_ML.name
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{name_lr}\n')


import NN_Model_UO
current_time_uo = NN_Model_UO.current_time
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{current_time_uo}\n')


import XGBoost_Model_UO
name_uo = XGBoost_Model_UO.name
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{name_uo}\n')

import Logistic_Regression_UO
name_lr_uo = Logistic_Regression_UO.name
with open(rf'C:/Users/antho/cursorProjects/NBA-Machine-Learning-Sports-Betting/models.txt', 'a') as the_file:
    the_file.write(f'{name_lr_uo}\n')



