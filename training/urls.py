from django.urls import path
from . import views

app_name = 'training'

urlpatterns = [
    path('', views.training_list, name='list'),
    path('new/', views.training_create, name='create'),
    path('<int:pk>/', views.training_detail, name='detail'),
    path('<int:pk>/retrain/', views.training_retrain, name='retrain'),
]
