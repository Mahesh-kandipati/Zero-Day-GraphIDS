from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_home, name='index'),
    path('embeddings/', views.embeddings_view, name='embeddings'),
    path('graph-explorer/', views.graph_explorer, name='graph_explorer'),
    path('system-status/', views.system_status, name='system_status'),
]
