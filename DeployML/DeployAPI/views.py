from .forms import TeamForm
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from .models import Teamset
from .serializer import TeamSerializers

import pickle
import json
import numpy as np
from sklearn import preprocessing
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages


class TeamView(viewsets.ModelViewSet):
    queryset = Teamset.objects.all()
    serializer_class = TeamSerializers


def status(df):
    try:
        model = pickle.load(open("/Users/HP/Django_machine_learning/DeployML/DeployAPI/TeamPredictions.sav", 'rb'))
        X = df
        y_pred = model.predict(X)
        result = "Win" if y_pred == 1 else "Loose"
        return result, y_pred
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def FormView(request):
    if request.method == 'POST':
        form = TeamForm(request.POST or None)

        if form.is_valid():
            GamesPlayed= form.cleaned_data['GamesPlayed']
            SumPointBefore = form.cleaned_data['SumPointBefore']
            SumGoalFor = form.cleaned_data['SumGoalFor']
            SumGoalAgainst = form.cleaned_data['SumGoalAgainst']
            countWining = form.cleaned_data['countWining']
            countLoose = form.cleaned_data['countLoose']
            countDraw = form.cleaned_data['countDraw']
            GoalDiff = form.cleaned_data['GoalDiff']


            df = pd.DataFrame({'GamesPlayed': [GamesPlayed],
                               'SumPointBefore': [SumPointBefore],
                               'SumGoalFor': [SumGoalFor],
                               'SumGoalAgainst': [SumGoalAgainst],
                               'countWining': [countWining],
                               'countLoose':[countLoose],
                               'countDraw': [countDraw],
                               'GoalDiff': [GoalDiff],
                               }, )
            result = status(df)
            return render(request, 'DeployAPI/status.html', {"data": result})

    form = TeamForm()
    return render(request, 'DeployAPI/form.html', {'form': form})