# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

"""
Wzór:
    p(k, m) = [m^k * e^(-m)] / k!
    gdzie:
        k = ilość goli
        m = średnia ilość goli
        e = liczba Eulera
"""

"""
Wzór2:

"""



"""
tabels = pd.read_html('https://annabet.com/pl/soccerstats/serie_219_Czech_2._Liga.html')
main = tabels[5]
main.index = main[0]
main = main[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
main = main.sort_values(by=1)
home = tabels[6]
home.index = home[0]
home = home[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
home = home.sort_values(by=1)
home.index = main.index
away = tabels[7]
away.index = away[0]
away = away[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
away = away.sort_values(by=1)
away.index = main.index

mean_force_goals_league = np.sum(pd.to_numeric(main[6])) / np.sum(pd.to_numeric(main[2]))
mean_force_goals_league_home = np.sum(pd.to_numeric(home[6])) / np.sum(pd.to_numeric(home[2]))
mean_force_goals_league_away = np.sum(pd.to_numeric(away[6])) / np.sum(pd.to_numeric(away[2]))
home['Off'] = pd.Series(pd.to_numeric(home[12])) / mean_force_goals_league_home
home['Deff'] = pd.Series(pd.to_numeric(home[13])) / mean_force_goals_league_away
away['Off'] = pd.Series(pd.to_numeric(away[12])) / mean_force_goals_league_away
away['Deff'] = pd.Series(pd.to_numeric(away[13])) / mean_force_goals_league_home
#predict_home_score = home.iloc[12, 13] * away.iloc[5, 14] * mean_force_goals_league_home
#predict_away_score = away.iloc[5, 13] * home.iloc[12, 14] * mean_force_goals_league_away
predict_home_score = home.loc[f'15.', 'Off'] * away.loc[f'14.', 'Deff'] * mean_force_goals_league_home
predict_away_score = away.loc[f'14.', 'Off'] * home.loc[f'15.', 'Deff'] * mean_force_goals_league_away

home_poisson = []
away_poisson = []
result = []
for i in [0, 1, 2, 3, 4, 5]:
    data = ((predict_home_score ** i) * (e ** (-1 * predict_home_score))) / math.factorial(i)
    home_poisson.append(round(data, 3))

for i in [0, 1, 2, 3, 4, 5]:
    data = ((predict_away_score ** i) * (e ** (-1 * predict_away_score))) / math.factorial(i)
    away_poisson.append(round(data, 3))


for el in home_poisson:
    data = []
    for value in away_poisson:
        var = el * value
        data.append(var)
    result.append(data)

print(home_poisson)
print(away_poisson)

print(home.loc[f'15.', 1], away.loc[f'14.', 1])
df_result = pd.DataFrame(result).transpose()
"""

"""
def poisson_predict(url=str, home_pos=int, away_pos=int):
    e = np.exp(1)
    page_url = url
    tabels = pd.read_html(url)

    main = tabels[5]
    main.index = main[0]
    main = main[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    main = main.sort_values(by=1)
    home = tabels[6]
    home.index = home[0]
    home = home[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    home = home.sort_values(by=1)
    home.index = main.index
    away = tabels[7]
    away.index = away[0]
    away = away[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    away = away.sort_values(by=1)
    away.index = main.index
    mean_force_goals_league = np.sum(pd.to_numeric(main[6])) / np.sum(pd.to_numeric(main[2]))
    mean_force_goals_league_home = np.sum(pd.to_numeric(home[6])) / np.sum(pd.to_numeric(home[2]))
    mean_force_goals_league_away = np.sum(pd.to_numeric(away[6])) / np.sum(pd.to_numeric(away[2]))
    home['Off'] = pd.Series(pd.to_numeric(home[12])) / mean_force_goals_league_home
    home['Deff'] = pd.Series(pd.to_numeric(home[13])) / mean_force_goals_league_away
    away['Off'] = pd.Series(pd.to_numeric(away[12])) / mean_force_goals_league_away
    away['Deff'] = pd.Series(pd.to_numeric(away[13])) / mean_force_goals_league_home
    predict_home_score = home.loc[f'{home_pos}.', 'Off'] * away.loc[
        f'{away_pos}.', 'Deff'] * mean_force_goals_league_home
    predict_away_score = away.loc[f'{away_pos}.', 'Off'] * home.loc[
        f'{home_pos}.', 'Deff'] * mean_force_goals_league_away
    home_poisson = []
    away_poisson = []
    print(main.loc[f'{home_pos}.', 1] + f' vs ' + main.loc[f'{away_pos}.', 1])
    result = []
    for i in [0, 1, 2, 3, 4, 5]:
        data = ((predict_home_score ** i) * (e ** (-1 * predict_home_score))) / math.factorial(i)
        home_poisson.append(round(data, 3))

    for i in [0, 1, 2, 3, 4, 5]:
        data = ((predict_away_score ** i) * (e ** (-1 * predict_away_score))) / math.factorial(i)
        away_poisson.append(round(data, 3))

    for el in home_poisson:
        data = []
        for value in away_poisson:
            var = el * value
            data.append(var)
        result.append(data)

    df_result1 = round(pd.DataFrame(result).transpose() * 100, 3)

    return df_result1
"""

def football_cor(url=str, home_pos=int, away_pos=int):
    my_function_url = url
    columns = ['Position', 'Team_name', 'Games_played', 'Wins', 'Draws', 'Loses', 'Goals_for', 'Goals_against',
               'Goals_different', 'Points', 'Point_per_games_played', 'Win_percents', 'Average_goals_for',
               'Average_goals_against']

    tables = pd.read_html(my_function_url)
    table_main = tables[5]
    table_main.columns = columns
    table_main.index = table_main.Position
    table_main = table_main[['Team_name', 'Games_played', 'Goals_for', 'Goals_against', 'Average_goals_for',
                             'Average_goals_against']]
    table_main = table_main[1:]
    table_main = table_main.sort_values(by='Team_name')
    series_std1_main = pd.to_numeric(table_main['Average_goals_for'])
    series_std2_main = pd.to_numeric(table_main['Average_goals_against'])
    my_std_main = (series_std1_main.std() + series_std2_main.std()) / 2
    table_home = tables[6]
    table_home.columns = columns
    table_home.index = table_home.Position
    table_home = table_home[['Team_name', 'Games_played', 'Goals_for', 'Goals_against', 'Average_goals_for',
                             'Average_goals_against']]
    table_home = table_home[1:]
    table_home = table_home.sort_values(by='Team_name')
    table_home.index = table_main.index
    table_away = tables[7]
    table_away.columns = columns
    table_away.index = table_away.Position
    table_away = table_away[['Team_name', 'Games_played', 'Goals_for', 'Goals_against', 'Average_goals_for',
                             'Average_goals_against']]
    table_away = table_away[1:]
    table_away = table_away.sort_values(by='Team_name')
    table_away.index = table_main.index
    series_std1_home_away = (pd.to_numeric(table_home['Average_goals_for']) + pd.to_numeric(
        table_away['Average_goals_for'])) / 2
    series_std2_home_away = (pd.to_numeric(table_home['Average_goals_against']) + pd.to_numeric(
        table_away['Average_goals_against'])) / 2
    my_std_home_away = (series_std1_home_away.std() + series_std2_home_away.std()) / 2
    home = [value for value in table_home.loc[f'{home_pos}.', :].values]
    away = [value for value in table_away.loc[f'{away_pos}.', :].values]
    dir_to_cor = {f'{home[0]}': [float(v) for v in home[1:]], f'{away[0]}': [float(v) for v in away[1:]]}
    df_to_cor = pd.DataFrame(dir_to_cor)
    home1 = [value for value in table_main.loc[f'{home_pos}.', :].values]
    away1 = [value for value in table_main.loc[f'{away_pos}.', :].values]
    dir_to_cor1 = {f'{home1[0]}': [float(v) for v in home1[1:]], f'{away1[0]}': [float(v) for v in away1[1:]]}
    df_to_cor1 = pd.DataFrame(dir_to_cor1)
    result = {'info1' : dir_to_cor, 'info2' : round((dir_to_cor[f'{home[0]}'][3] + dir_to_cor[f'{home[0]}'][4] +
                                                     dir_to_cor[f'{away[0]}'][3] + dir_to_cor[f'{away[0]}'][4]) / 2, 3),
              'info3' : round((dir_to_cor[f'{home[0]}'][3] + dir_to_cor[f'{away[0]}'][3]) / 2, 3),
              'info4' : round((dir_to_cor[f'{home[0]}'][4] + dir_to_cor[f'{away[0]}'][4]) / 2, 3),
              'info5' : round(my_std_home_away, 4), 'info6' : dir_to_cor1,
              'info7' : round((dir_to_cor1[f'{home1[0]}'][3] + dir_to_cor1[f'{home1[0]}'][4] + dir_to_cor1[f'{away1[0]}'][3] +
                 dir_to_cor1[f'{away1[0]}'][4]) / 2, 3),
              'info8' : round((dir_to_cor1[f'{home1[0]}'][3] + dir_to_cor1[f'{away1[0]}'][3]) / 2, 3),
              'info9' : round((dir_to_cor1[f'{home1[0]}'][4] + dir_to_cor1[f'{away1[0]}'][4]) / 2, 3),
              'info10' : round(my_std_main, 4), 'info11' : df_to_cor.corr(method='pearson').to_html(),
              'info12' : df_to_cor1.corr(method='pearson').to_html()}
    return result

'''
(dir_to_cor, f'\nŚrednia goli na mecz:', round((dir_to_cor[f'{home[0]}'][3] +
                                                         dir_to_cor[f'{home[0]}'][4] + dir_to_cor[f'{away[0]}'][3] +
                                                         dir_to_cor[f'{away[0]}'][4]) / 2, 3),
          f'| Strzelone:', round((dir_to_cor[f'{home[0]}'][3] + dir_to_cor[f'{away[0]}'][3]) / 2, 3),
          f'| Stracone:', round((dir_to_cor[f'{home[0]}'][4] + dir_to_cor[f'{away[0]}'][4]) / 2, 3),
          f'| Odchylenie: {round(my_std_home_away, 4)}\n', dir_to_cor1, f'\nŚrednia goli na mecz:',
          round((dir_to_cor1[f'{home1[0]}'][3] + dir_to_cor1[f'{home1[0]}'][4] + dir_to_cor1[f'{away1[0]}'][3] +
                 dir_to_cor1[f'{away1[0]}'][4]) / 2, 3), f'| Strzelone:', round((dir_to_cor1[f'{home1[0]}'][3] +
                                                                                 dir_to_cor1[f'{away1[0]}'][3]) / 2,
                                                                                3), f'| Stracone:',
          round((dir_to_cor1[f'{home1[0]}'][4] +
                 dir_to_cor1[f'{away1[0]}'][4]) / 2, 3), f'| Odchylenie: {round(my_std_main, 4)}\n',
          f"\nKorelacja dla tabel home i away\n", f"{df_to_cor.corr(method='pearson')}",
          f"\nKorelacja dla tabeli main\n{df_to_cor1.corr(method='pearson')}")'''


def poisson_predict(url=str, home_pos=int, away_pos=int):
    e = np.exp(1)
    tabels = pd.read_html(url)

    main = tabels[5]
    main.index = main[0]
    main = main[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    main = main.sort_values(by=1)
    home = tabels[6]
    home.index = home[0]
    home = home[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    home = home.sort_values(by=1)
    home.index = main.index
    away = tabels[7]
    away.index = away[0]
    away = away[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]][1:]
    away = away.sort_values(by=1)
    away.index = main.index
    mean_force_goals_league = np.sum(pd.to_numeric(main[6])) / np.sum(pd.to_numeric(main[2]))
    mean_force_goals_league_home = np.sum(pd.to_numeric(home[6])) / np.sum(pd.to_numeric(home[2]))
    mean_force_goals_league_away = np.sum(pd.to_numeric(away[6])) / np.sum(pd.to_numeric(away[2]))
    home['Off'] = pd.Series(pd.to_numeric(home[12])) / mean_force_goals_league_home
    home['Deff'] = pd.Series(pd.to_numeric(home[13])) / mean_force_goals_league_away
    away['Off'] = pd.Series(pd.to_numeric(away[12])) / mean_force_goals_league_away
    away['Deff'] = pd.Series(pd.to_numeric(away[13])) / mean_force_goals_league_home
    predict_home_score = home.loc[f'{home_pos}.', 'Off'] * away.loc[
        f'{away_pos}.', 'Deff'] * mean_force_goals_league_home
    predict_away_score = away.loc[f'{away_pos}.', 'Off'] * home.loc[
        f'{home_pos}.', 'Deff'] * mean_force_goals_league_away
    home_poisson = []
    away_poisson = []
    print(main.loc[f'{home_pos}.', 1] + f' vs ' + main.loc[f'{away_pos}.', 1])
    result = []
    for i in [0, 1, 2, 3, 4, 5]:
        data = ((predict_home_score ** i) * (e ** (-1 * predict_home_score))) / math.factorial(i)
        home_poisson.append(round(data, 3))

    for i in [0, 1, 2, 3, 4, 5]:
        data = ((predict_away_score ** i) * (e ** (-1 * predict_away_score))) / math.factorial(i)
        away_poisson.append(round(data, 3))

    for el in home_poisson:
        data = []
        for value in away_poisson:
            var = el * value
            data.append(round(var, 4))
        result.append(data)

    df_result1 = pd.DataFrame(result).transpose() * 100

    values = df_result1.values

    my_values = []

    for i in values:
        for el in i:
            my_values.append(el)

    my_values.sort(reverse=True)

    my_values = my_values[:4]

    def style_func(v, value, other, list=my_values):

        cond = v in list

        return value if cond else other

    final_result = df_result1.style.applymap(style_func, value='color:green;', other='color:red;')

    return final_result