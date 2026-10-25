from django.urls import path
from . import views

app_name = 'detection'

urlpatterns = [
    path('upload/', views.upload_traffic, name='upload'),
    path('result/<int:pk>/', views.detection_result, name='result'),
    path('history/', views.detection_history, name='history'),
    path('alerts/', views.alert_list, name='alerts'),
    path('alerts/<int:pk>/update/', views.alert_update_status, name='alert_update'),
]
