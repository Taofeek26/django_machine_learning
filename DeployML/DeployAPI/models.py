from django.db import models

# Create your models here.

class Teamset(models.Model):
    GamesPlayed = models.IntegerField()
    SumPointBefore = models.IntegerField()
    SumGoalFor = models.IntegerField()
    SumGoalAgainst = models.IntegerField()
    countWining = models.IntegerField()
    countLoose = models.IntegerField()
    countDraw = models.IntegerField()
    GoalDiff = models.IntegerField()
    def __str__(self):
        return self.Team
