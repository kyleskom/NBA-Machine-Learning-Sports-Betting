

# Author: Neal Mick
# Created: November 2023
# nbadata.cloud nealmick.com




    


import pandas as pd
import numpy as np
import requests,pickle,time
from datetime import datetime, timedelta

labels = ['ast','blk','dreb','fg3_pct','fg3a','fg3m','fga','fgm','fta','ftm','oreb','pf','pts','reb','stl', 'turnover', 'min']

#save pickle object file
def save_obj(obj, name):
    with open('../../Data/objects/' + name + '.pkl', 'wb') as f:  # Change 'rb' to 'wb'
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#loads pickle object file
def load_obj(name):
    with open('../../Data/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# the relative path changes based on if your running future predictions the path is from main.py file
# but when training the path needs ../../ to find objects soo we have 2 of these currently
def load_obj_root(name):
    with open('Data/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def save_obj_root(obj, name):
    with open('Data/objects/' + name + '.pkl', 'wb') as f:  # Change 'rb' to 'wb'
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#request url
def req(url):
    print('requesting url: '+url)
    r = requests.get(url)#request url
    if str(r) != '<Response [200]>':#got bad reply
        time.sleep(5)#wait 5 seconds,although server shuts down for a bit if rate limit was hit
        req(url)
    print('success')
    time.sleep(2)#lets rate limit a bit here
    return r.json()






def formLastGame(data,team):
    #print('forming last game data')
    if data is None:
        print('no data=---------=========------------==========')

    if data['data'][0]['game']['home_team_id'] == team:
        history_id = data['data'][0]['game']['home_team_id']
        opponent_id = data['data'][0]['game']['visitor_team_id']
        history_score = data['data'][0]['game']['home_team_score']
        opponent_score = data['data'][0]['game']['visitor_team_score']
    else:
        opponent_id = data['data'][0]['game']['home_team_id']
        history_id = data['data'][0]['game']['visitor_team_id']
        opponent_score = data['data'][0]['game']['home_team_score']
        history_score = data['data'][0]['game']['visitor_team_score']
    gameid = data['data'][0]['game']['id']

    opponent_players = []
    history_players = []
    
    for player in  data['data']:
        p = {}
        p['id'] = player['id']
        p['teamid'] = player['team']['id']
        for label in labels:
            if player['min'] is None:
                continue
            if label == 'min':
                min = player['min']
                min = min.split(':')[0]
                player['min']=min
            p[label] = player[label]
        if player['min'] is None:
            #print('min is none------------')
            #the player has no play time for this game....
            continue
        if p['teamid'] == history_id:
            history_players.append(p)
        if p['teamid'] == opponent_id:
            opponent_players.append(p)

    best_history_players = []
    best_opponent_players = []

    for i in range(0,5):
        best = historyBestPlayer(history_players)
        if best == '':
            break
        best_player = history_players.pop(int(best))
        best_history_players.append(best_player)


    for i in range(0,5):
        best = historyBestPlayer(opponent_players)
        if best == '':
            break
        best_player = opponent_players.pop(int(best))
        best_opponent_players.append(best_player)

    r = {
        'best_history_players': best_history_players,
        'best_opponent_players': best_opponent_players,
        'gameid': gameid,
        'history_score': history_score,
        'opponent_score': opponent_score
    }

    
    return r


#we decide on the best player based on minutes played
def historyBestPlayer(players):
    best = ''
    topMin = 0
    for player in range(len(players)):
        min = players[player]['min']
        min = min.split(':')[0]
        #print(min,topMin)
        if min == '':
            continue
        if int(min) > int(topMin):
            best = player
            topMin = min
    return best





def getLastGame(game_date,team_id,cache,game_objects):

    # Calculate dates for one month and one day before the game
    one_month_before_game = (game_date - timedelta(days=30)).strftime('%Y-%m-%d')
    one_day_before_game = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
    #grab the year as a int for getting the game season later
    year = int(game_date.strftime('%Y'))


    # games endpoint url with params for start date, end date and team id...
    # this api call basically gets the last month of games before the game date...
    # we use the url as our cache key in order to make things go alot faster...
    url = 'https://www.balldontlie.io/api/v1/games?start_date=' + one_month_before_game + '&end_date=' + one_day_before_game + '&&team_ids[]=' + str(team_id) + '&per_page=100'   
    cached = False
    try:
        r = cache[url]
        cached = True
    except KeyError:
        print('no cached data')
        r = req(url)
        r = r['data']
        r.reverse()
        cache[url] = r
        save_obj(cache,'cache')

    # we iterate over the games in the response data to find closest...
    closest_game = None
    min_diff = float('inf')  # Start with a very large number
    for game in r:
        #print(game['id'],game['date'],game['home_team_score'],game['visitor_team_score'],game['status'])
        if game['status'] == 'Final':
            game_datetime = datetime.strptime(game['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            diff = abs((game_datetime - game_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_game = game
                lastID = closest_game['id']
                #print('found close game date setting game id here to: '+str(lastID))
                #print(game_date.strftime('%Y-%m-%d'),game['date'])
   
    try:
        print(lastID)
    except UnboundLocalError:
        print('no last id, returning none')
        return None
    
    # basically here we get the current season labeled as the game data year
    # To avoid any issues just in case we also get the season before and after
    # we check for the last game id in all 3 seasons in order to get max coverage
    try:
        g1 = game_objects[str(year)]
        g2 = game_objects[str(year-1)]
        g3 = game_objects[str(year+1)]
    except KeyError:
        return 'out-of-data'#means we  got no more data
    try:
        g = g1[lastID]
    except KeyError:
       pass
    try:
        g = g2[lastID]
    except KeyError:
        pass
    try:
        g = g3[lastID]
    except KeyError:
        pass
    try:
        if g is None:
            return None
    except UnboundLocalError:
        return None
    

    data = formLastGame(g, team_id)

    #print('got last game data!!')
    formed_data = []
    formed_data.append(data['history_score'])
    formed_data.append(data['opponent_score'])

    for player in data['best_history_players']:
        for label in labels:
            formed_data.append(player[label])

    for player in data['best_opponent_players']:
        for label in labels:
            formed_data.append(player[label])

    return formed_data






def AugmentData(data):
    # can add or remove this line here, only works if its been run once before and we have cache saved
    # but makes it alot faster if re running

    return load_obj('res')#uncomment here to load from cache

    header = getLabels()
    for stat in header:
        data[stat] = None  # Initialize new columns


    cache = load_obj('cache')
    team_name_ids = load_obj('teamNamesById')
    id_team_names = {}
    for id in team_name_ids:
        id_team_names[team_name_ids[id]] = id

    game_objects = {}
    for i in range(2009,2023):
        print('loading games for year: '+str(i))
        games = load_obj(str(i)+'Games')
        game_objects[str(i)] = games


    for foo in range(len(data)):


        ## some of the team names dont match in the data set and the api
        if data['TEAM_NAME'][foo] == 'Los Angeles Clippers':
            data['TEAM_NAME'][foo] = 'LA Clippers'
        if data['TEAM_NAME.1'][foo] == 'Los Angeles Clippers':
            data['TEAM_NAME.1'][foo] = 'LA Clippers'

        if data['TEAM_NAME'][foo] == 'Charlotte Bobcats':
            data['TEAM_NAME'][foo] = 'Charlotte Hornets'
        if data['TEAM_NAME.1'][foo] == 'Charlotte Bobcats':
            data['TEAM_NAME.1'][foo] = 'Charlotte Hornets'

        if data['TEAM_NAME'][foo] == 'New Orleans Hornets':
            data['TEAM_NAME'][foo] = 'New Orleans Pelicans'
        if data['TEAM_NAME.1'][foo] == 'New Orleans Hornets':
            data['TEAM_NAME.1'][foo] = 'New Orleans Pelicans'


        # this converts the correct team names to team ids which we use in the api
        home_team_id = id_team_names[data['TEAM_NAME'][foo]]
        visitor_team_id = id_team_names[data['TEAM_NAME.1'][foo]]
        
        
        # Convert string to date object
        game_date = datetime.strptime(data['Date'][foo], '%Y-%m-%d')
        home_team_lg_data = getLastGame(game_date,home_team_id,cache,game_objects)
        visitor_team_lg_data = getLastGame(game_date,visitor_team_id,cache,game_objects)
       
        if home_team_lg_data == 'out-of-data' or visitor_team_lg_data == 'out-of-data':
            print('out of data')
            #this is the end of the training data script. we should exit here ideally if things are working.
            break
       
        # Concatenate home and visitor data
        try:
            combined_data = home_team_lg_data + visitor_team_lg_data
        except TypeError:
            data = data.drop(foo)
            print('type error')
            #we just skip rows we dont want
            continue

        # Assign all data for this row at once
        data.loc[foo, header[:len(combined_data)]] = combined_data

        print('Done with row:', foo)
        '''
        print(data.iloc[foo].to_string())  # Display the updated row
        print(visitor_team_lg_data[0],visitor_team_lg_data[1])
        print(home_team_lg_data[0],home_team_lg_data[1])
        print(header)
        '''
    #drop any rows with values that didnt get filled in...
    data = data.dropna()

    print(data.iloc[0].to_string())  # Display the first updated row
    print(data.iloc[-1].to_string())  # Display the last updated row

    save_obj(data,'res')#save for late, makes it alot faster if re running over and over...
    return data







#this function generates the header labels (column names_
def getLabels():
    header = []
    header.append('home_history_score')
    header.append('home_history_op_score')
    for i in range(0,5):
        for label in labels:
            header.append('hh-'+str(i)+'-'+label)
    for i in range(0,5):
        for label in labels:
            header.append('ho-'+str(i)+'-'+label)
    header.append('visitor_history_score')
    header.append('visitor_history_op_score')
    for i in range(0,5):
        for label in labels:
            header.append('vh-'+str(i)+'-'+label)
    for i in range(0,5):
        for label in labels:
            header.append('vo-'+str(i)+'-'+label)
    headers = ''
    for stat in header:
        headers += stat+','
    return header



def AugmentFutureData(home_team, away_team,row):
    print('Augmenting data for',home_team, 'v',away_team)
    home_team_lg_data = getLastGameFutures(home_team)
    visitor_team_lg_data = getLastGameFutures(away_team)
    combined_data = home_team_lg_data + visitor_team_lg_data
    combined_data = np.array(combined_data, dtype=np.float32)
    row = np.concatenate((row, combined_data), axis=0)
    return row





def getLastGameFutures(team_name):
    print('getting last game for: '+team_name)
    team_name_ids = load_obj_root('teamNamesById')
    id_team_names = {}
    for id in team_name_ids:
        id_team_names[team_name_ids[id]] = id
    team_id = id_team_names[team_name]


    game_date = datetime.now()
    one_month_before_game = (game_date - timedelta(days=30)).strftime('%Y-%m-%d')
    one_day_before_game = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
    #grab the year as a int for getting the game season later
    year = int(game_date.strftime('%Y'))


    # games endpoint url with params for start date, end date and team id...
    # this api call basically gets the last month of games before the game date...
    # we use the url as our cache key in order to make things go alot faster...
    #save_obj_root({},'futureCache')#you can clear a cache by doing this too
   
    futureCache = load_obj_root('futureCache')

    url = 'https://www.balldontlie.io/api/v1/games?start_date=' + one_month_before_game + '&end_date=' + one_day_before_game + '&&team_ids[]=' + str(team_id) + '&per_page=100'   

    try:
        r = futureCache[url]
    except KeyError:
        r = req(url)
        r = r['data']
        r.reverse()
        futureCache[url] = r
        save_obj_root(futureCache,'futureCache')
    # we iterate over the games in the response data to find closest...
    closest_game = None
    min_diff = float('inf')  # Start with a very large number
    for game in r:
        #print(game['id'],game['date'],game['home_team_score'],game['visitor_team_score'],game['status'])
        if game['status'] == 'Final':
            game_datetime = datetime.strptime(game['date'], '%Y-%m-%dT%H:%M:%S.%fZ')
            diff = abs((game_datetime - game_date).days)
            if diff < min_diff:
                min_diff = diff
                closest_game = game
                lastID = closest_game['id']
                #print('found close game date setting game id here to: '+str(lastID))
                #print(game_date.strftime('%Y-%m-%d'),game['date'])


    url = 'https://www.balldontlie.io/api/v1/stats?game_ids[]='+str(lastID)

    try:
        r = futureCache[url]
        print(url)
        print('found in cache')

    except KeyError:
        r = req(url)
        futureCache[url] = r
        save_obj_root(futureCache,'futureCache')

    data =  formLastGame(r,team_id)
    formed_data = []
    formed_data.append(data['history_score'])
    formed_data.append(data['opponent_score'])

    for player in data['best_history_players']:
        for label in labels:
            formed_data.append(player[label])

    for player in data['best_opponent_players']:
        for label in labels:
            formed_data.append(player[label])

    return formed_data















