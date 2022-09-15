from rest_framework import serializers
from .models import Teamset

class TeamSerializers(serializers.ModelSerializer):
    class meta:
        model = Teamset
        fields = '__all__'
