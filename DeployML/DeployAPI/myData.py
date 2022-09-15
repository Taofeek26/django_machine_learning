import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import pickle
#% matplotlibinline

df = pd.read_csv('results.csv')

Team = input("Enter the name of the team: ")
Team = Team.title()

arsenal_table = df[(df['home_team'] == Team) | (df['away_team'] == Team)]

arsenal_table['points_before'] = np.nan


def impute_column_data(cols):
    points_before = cols[0]
    result = cols[1]
    home_team = cols[2]
    away_team = cols[3]
    season = cols[4]

    while season:
        if pd.isnull(points_before):

            if result == 'D':
                return 1

            elif (result == 'H') & (home_team == Team):
                return 3

            elif (result == 'A') & (away_team == Team):
                return 3

            else:
                return 0

        else:
            return 0


arsenal_table['points_before'] = arsenal_table[['points_before','result',
                                                'home_team', 'away_team','season']].apply(impute_column_data, axis = 1)

arsenal_table['Sum_Points_before'] = np.nan

sessions = arsenal_table['season'].unique()

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        w = 0
        l = []
        for i in y:
            l.append(w)
            w = w + i
        return l


    arsenal_table['Sum_Points_before'][arsenal_table['season'] == session] = sum_(sum_df['points_before'])

arsenal_table[['Goal_For', 'Goal_Against']] = np.nan, np.nan


def impute_column_data(cols):
    Goal_Against = cols[0]
    home_team = cols[1]
    away_team = cols[2]
    season = cols[3]
    home_goal = cols[4]
    away_goal = cols[5]

    while season:
        if pd.isnull(Goal_Against):

            if home_team != Team:
                return home_goal


            elif away_team != Team:
                return away_goal

            else:
                return 0

        else:
            return 0


arsenal_table['Goal_Against'] = arsenal_table[['Goal_Against', 'home_team', 'away_team',
                                               'season', 'home_goals', 'away_goals']].apply(impute_column_data, axis=1)


def impute_column_data(cols):
    Goal_For = cols[0]
    home_team = cols[1]
    away_team = cols[2]
    season = cols[3]
    home_goal = cols[4]
    away_goal = cols[5]

    while season:
        if pd.isnull(Goal_For):

            if home_team == Team:
                return home_goal


            elif away_team == Team:
                return away_goal

            else:
                return 0

        else:
            return 0


arsenal_table['Goal_For'] = arsenal_table[['Goal_For', 'home_team', 'away_team',
                                           'season', 'home_goals', 'away_goals']].apply(impute_column_data, axis=1)

arsenal_table['Sum_Goal_For'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        w = 0
        l = []
        for i in y:
            l.append(w)
            w = w + i
        return l


    arsenal_table['Sum_Goal_For'][arsenal_table['season'] == session] = sum_(sum_df['Goal_For'])

arsenal_table['Sum_Goal_Against'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        w = 0
        l = []
        for i in y:
            l.append(w)
            w = w + i
        return l


    arsenal_table['Sum_Goal_Against'][arsenal_table['season'] == session] = sum_(sum_df['Goal_Against'])

arsenal_table['Result'] = np.nan


def impute_column_data(cols):
    Result = cols[0]
    result = cols[1]
    home_team = cols[2]
    away_team = cols[3]
    season = cols[4]

    while season:
        if pd.isnull(Result):

            if result == 'D':
                return 'Draw'

            elif (result == 'H') & (home_team == Team):
                return 'Win'

            elif (result == 'A') & (away_team == Team):
                return 'Win'

            else:
                return 'Loose'

        else:
            return 'Loose'


arsenal_table['Result'] = arsenal_table[['Result', 'result', 'home_team',
                                         'away_team', 'season']].apply(impute_column_data, axis=1)

arsenal_table['count_wining'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        l = []
        gf = []
        for x in y:
            if x == 'Win':
                l.append(x)
                s = len(l)
                gf.append(s)
            else:
                gf.append(0)
        gf.insert(0, 0)
        gf.pop()
        return gf


    arsenal_table['count_wining'][arsenal_table['season'] == session] = sum_(sum_df['Result'])

arsenal_table['count_loose'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        l = []
        gf = []
        for x in y:
            if x == 'Loose':
                l.append(x)
                s = len(l)
                gf.append(s)
            else:
                gf.append(0)
        gf.insert(0, 0)
        gf.pop()
        return gf


    arsenal_table['count_loose'][arsenal_table['season'] == session] = sum_(sum_df['Result'])

arsenal_table['count_draw'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]


    def sum_(y):
        l = []
        gf = []
        for x in y:
            if x == 'Draw':
                l.append(x)
                s = len(l)
                gf.append(s)
            else:
                gf.append(0)
        gf.insert(0, 0)
        gf.pop()
        return gf


    arsenal_table['count_draw'][arsenal_table['season'] == session] = sum_(sum_df['Result'])

arsenal_table['Goal_diff'] = np.nan


def impute_column_data(cols):
    Goal_diff = cols[0]
    Sum_Goal_For = cols[1]
    Sum_Goal_Against = cols[2]
    season = cols[3]

    while season:
        if pd.isnull(Goal_diff):
            gd = Sum_Goal_For - Sum_Goal_Against
            return gd

        else:
            return 'Loose'


arsenal_table['Goal_diff'] = arsenal_table[['Goal_diff', 'Sum_Goal_For',
                                            'Sum_Goal_Against', 'season']].apply(impute_column_data, axis=1)

arsenal_table['number_of games'] = np.nan

for session in sessions:
    sum_df = arsenal_table[arsenal_table['season'] == session]
    arsenal_table['number_of games'][arsenal_table['season'] == session] = np.arange(0, len(sum_df))

arsenal_table.set_index('number_of games', inplace=True)

arsenal_table.reset_index(inplace=True)

Result = pd.get_dummies(arsenal_table['Result'])
result = pd.get_dummies(arsenal_table['result'])

Arsenal_Data = pd.concat([arsenal_table, Result, result], axis=1)

Arsenal_Data.drop(['Result', 'points_before', 'Goal_For', 'Goal_Against',
                   'season', 'home_team', 'away_team', 'result'], axis=1, inplace=True)

Arsenal_Data.rename(columns={'number_of games': 'GamesPlayed',
                             'home_goals': 'HomeGoals',
                             'away_goals': 'AwayGoals',
                             'Sum_Points_before': 'SumPointBefore',
                             'Sum_Goal_For': 'SumGoalFor',
                             'Sum_Goal_Against': 'SumGoalAgainst',
                             'count_wining': 'countWining',
                             'count_loose': 'countLoose',
                             'count_draw': 'countDraw',
                             'Goal_diff': 'GoalDiff', },
                    inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    Arsenal_Data.drop(['Win', 'HomeGoals', 'AwayGoals', 'A', 'D', 'H', 'Draw', 'Loose'], axis=1),
    Arsenal_Data['Win'], test_size=0.30, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

file = "TeamPredictions.sav"
pickle.dump(logmodel, open(file, 'wb'))

predictions = logmodel.predict(X_test)
print(Team)
print(classification_report(y_test,predictions))
