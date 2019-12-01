from src import tools
import os
from tqdm import tqdm

url = 'https://stats.nba.com/stats/' \
      'leaguedashteamstats?Conference=&' \
      'DateFrom=10%2F29%2F{2}&DateTo={0}%2F{1}%2F{3}' \
      '&Division=&GameScope=&GameSegment=&LastNGames=0&' \
      'LeagueID=00&Location=&MeasureType=Base&Month=0&' \
      'OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&' \
      'PerMode=PerGame&Period=0&PlayerExperience=&' \
      'PlayerPosition=&PlusMinus=N&Rank=N&' \
      'Season={4}' \
      '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
      'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

year = [2007,2008, 2009, 2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
season = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
          "2017-18", "2018-19", "2019-20"]
month = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

begin_year_pointer = year[0]
end_year_pointer = year[0]
count = 0
year_count = 0

for season1 in tqdm(season):
      for month1 in tqdm(month):
            if month1 == 1:
                  count += 1
                  end_year_pointer = year[count]
            for day1 in tqdm(days):
                  try:
                        general_data = tools.get_json_data(url.format(month1, day1, begin_year_pointer, end_year_pointer, season1))
                        general_df = tools.to_data_frame(general_data)
                        general_df['Date'] = str(month1) + '-' + str(day1) + '-' + season1

                        directory2 = os.fsdecode('Team-Data')
                        name = directory2 + '/' +str(month1) + '-' + str(day1) + '-' + season1 + '.xlsx'
                        general_df.to_excel(name)
                  except:
                        continue
      year_count += 1
      begin_year_pointer = year[count]
