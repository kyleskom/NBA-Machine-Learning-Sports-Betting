import pandas as pd
import os

directory = os.fsdecode('Odds-Data')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    year = filename[9:-5]
    if filename.endswith('.xlsx'):
        df = pd.read_excel(directory + '/' + filename)
        x = pd.DataFrame(columns=['Date', 'Home', 'Away', 'OU', 'Spread', 'ML_Home', 'ML_Away', 'Points', 'Win_Margin'])
        count = 2
        date = ''
        home = ''
        away = ''
        ou = ''
        spread = ''
        ml_home = ''
        ml_away = ''
        points = ''
        margin = ''
        for row in df.itertuples():
            if count % 2 == 0:
                date = str(year + '-' + str(row[1]))
                away = str(row[4])
                if row[10] == 'pk':
                    ou = 0
                else:
                    ou = row[10]
                points = row[9]
                ml_away = str(row[12])
                count += 1
            else:
                home = str(row[4])
                if row[10] == 'pk':
                    spread = 0
                else:
                    spread = row[10]
                if spread > 50:
                    temp = spread
                    spread = ou
                    ou = temp
                ml_home = str(row[12])
                margin = row[9] - points
                points += row[9]
                temp = {
                    'Date': date,
                    'Home': home,
                    'Away': away,
                    'OU': ou,
                    'Spread': spread,
                    'ML_Home': ml_home,
                    'ML_Away': ml_away,
                    'Points': points,
                    'Win_Margin': margin
                }
                x = x.append(temp, ignore_index=True)
                count += 1
                date = ''
                home = ''
                away = ''
                ou = ''
                spread = ''
                ml_home = ''
                ml_away = ''
                points = ''
                margin = ''
        directory2 = os.fsdecode('Odds-Data-Clean')
        name = directory2 + '/' + year + '.xlsx'
        x.to_excel(name)
