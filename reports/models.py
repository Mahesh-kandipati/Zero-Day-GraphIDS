from django.db import models
from django.conf import settings
from training.models import TrainingJob
from detection.models import TrafficUpload

class Report(models.Model):
    TYPE_TRAINING = 'TRAINING'
    TYPE_DETECTION = 'DETECTION'
    TYPE_AUDIT = 'AUDIT'

    TYPE_CHOICES = [
        (TYPE_TRAINING, 'GNN Training & Evaluation Forensic Report'),
        (TYPE_DETECTION, 'Network Traffic Zero-Day Detection Report'),
        (TYPE_AUDIT, 'System Security Audit & Anomaly Export'),
    ]

    FORMAT_PDF = 'PDF'
    FORMAT_CSV = 'CSV'
    FORMAT_JSON = 'JSON'

    FORMAT_CHOICES = [
        (FORMAT_PDF, 'Adobe Portable Document Format (.pdf)'),
        (FORMAT_CSV, 'Comma Separated Values (.csv)'),
        (FORMAT_JSON, 'Structured Data (.json)'),
    ]

    title = models.CharField(max_length=250)
    report_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default=TYPE_DETECTION)
    file_format = models.CharField(max_length=10, choices=FORMAT_CHOICES, default=FORMAT_PDF)
    file = models.FileField(upload_to='reports/')

    training_job = models.ForeignKey(TrainingJob, on_delete=models.SET_NULL, null=True, blank=True, related_name='reports')
    traffic_upload = models.ForeignKey(TrafficUpload, on_delete=models.SET_NULL, null=True, blank=True, related_name='reports')

    generated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='generated_reports')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} [{self.file_format}] ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
