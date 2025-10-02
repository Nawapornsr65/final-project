from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),                # << หน้าแรก
    path("api/transcribe/", views.transcribe, name="transcribe"),
]
