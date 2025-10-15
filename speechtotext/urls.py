# speechtotext/urls.py
from django.urls import path
from . import views   # <<<<< สำคัญ

urlpatterns = [
    path("", views.index, name="index"),
    path("api/speech-to-text/", views.speech_to_text, name="speech_to_text"),
]
