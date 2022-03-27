from django.db import models

# Create your models here.
from django.urls import reverse


class League(models.Model):
    name = models.CharField(max_length=50, null=False, verbose_name='Liga')
    url = models.URLField(verbose_name='Link')
    country = models.CharField(max_length=32, null=False, verbose_name='Kraj')
    level = models.PositiveIntegerField(null=False, verbose_name='Poziom')

    def __str__(self):
        return f'{self.name}'

    def get_absolute_url(self):
        return reverse('current_league', kwargs={'pk' : self.pk})