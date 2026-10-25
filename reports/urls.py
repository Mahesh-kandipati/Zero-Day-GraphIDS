from django.urls import path
from . import views

app_name = 'reports'

urlpatterns = [
    path('', views.report_list, name='list'),
    path('generate/training/<int:job_id>/', views.generate_training_report_view, name='generate_training'),
    path('generate/detection/<int:upload_id>/', views.generate_detection_report_view, name='generate_detection'),
    path('download/<int:pk>/', views.download_report, name='download'),
    path('export/csv/<int:upload_id>/', views.export_scan_csv, name='export_csv'),
    path('export/json/<int:upload_id>/', views.export_scan_json, name='export_json'),
]
