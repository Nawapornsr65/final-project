from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),  # หน้าแรก
    path("api/speech-to-text/", views.speech_to_text, name="speech_to_text"),
]
