from django.http import HttpResponse
from django.shortcuts import render

from poisson_predict.functions import poisson_predict, football_cor
from poisson_predict.models import League
import pandas as pd


# Create your views here.



def main(request):
    return render(request, 'index.html')


def leagues(request):
    leagues = League.objects.all()

    return render(request, 'leagues.html', {'leagues' : leagues})


def current_league(request, pk):
    if request.method == 'GET':
        current_league = League.objects.get(pk=pk)
        tables = pd.read_html(f'{current_league.url}', header=0)
        table = tables[5]
        teams = table.iloc[:, 1]
        table.index = table.index + 1
        table_html_main = table.iloc[:, 1:].to_html()
        table = tables[6]
        table.index = table.index + 1
        table_html_home = table.iloc[:, 1:].to_html()
        table = tables[7]
        table.index = table.index + 1
        table_html_away = table.iloc[:, 1:].to_html()
        return render(request, 'current_league.html', {'current_league' : current_league,
                                                       'table_html_main' : table_html_main,
                                                       'table_html_home': table_html_home,
                                                       'table_html_away': table_html_away,
                                                       'teams' : teams})
    if request.method == 'POST':
        home = request.POST.get('Home')
        away = request.POST.get('Away')
        method = request.POST.get('Method')
        if method == 'corr':
            league = League.objects.get(pk=pk)
            corr = football_cor(f'{league.url}', int(home), int(away))
            return render(request, 'corr.html', {'message': f'Na razie działam', 'corr': corr})
        if method == 'poisson':
            league = League.objects.get(pk=pk)
            predict = poisson_predict(f'{league.url}', int(home), int(away))

            predict = predict.to_html()
            return render(request, 'poisson.html', {'message': f'Na razie działam', 'predict': predict})


def corr_describe(request):
    return render(request, 'corr_describe.html')


def poisson_describe(request):
    return render(request, 'poisson_describe.html')

