from django import forms
from .models import Teamset

class TeamForm(forms.ModelForm):
    class Meta:
              model = Teamset
              fields = "__all__"

    GamesPlayed = forms.IntegerField()
    SumPointBefore = forms.IntegerField()
    SumGoalFor = forms.IntegerField()
    SumGoalAgainst = forms.IntegerField()
    countWining = forms.IntegerField()
    countLoose = forms.IntegerField()
    countDraw = forms.IntegerField()
    GoalDiff = forms.IntegerField()