from django.db import models
from django.conf import settings
from training.models import TrainingJob

class TrafficUpload(models.Model):
    STATUS_PENDING = 'PENDING'
    STATUS_ANALYZING = 'ANALYZING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Queued for Analysis'),
        (STATUS_ANALYZING, 'Graph Inference Running'),
        (STATUS_COMPLETED, 'Inspection Completed'),
        (STATUS_FAILED, 'Analysis Failed'),
    ]

    VERDICT_BENIGN = 'BENIGN'
    VERDICT_SUSPICIOUS = 'SUSPICIOUS'
    VERDICT_HIGH_RISK = 'HIGH_RISK'
    VERDICT_ZERO_DAY = 'ZERO_DAY_ALERT'

    VERDICT_CHOICES = [
        (VERDICT_BENIGN, 'Benign Traffic - Normal Operations'),
        (VERDICT_SUSPICIOUS, 'Suspicious Traffic - Low Anomaly Density'),
        (VERDICT_HIGH_RISK, 'High Risk Intrusion - Elevated Anomaly Density'),
        (VERDICT_ZERO_DAY, 'Zero-Day Alert - Novel Attack Pattern Detected'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='traffic_scans')
    file = models.FileField(upload_to='traffic_uploads/')
    file_name = models.CharField(max_length=255)
    model_used = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='detections')

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    total_chunks = models.PositiveIntegerField(default=0)
    malicious_chunks = models.PositiveIntegerField(default=0)
    normal_chunks = models.PositiveIntegerField(default=0)
    avg_anomaly_score = models.FloatField(default=0.0)
    max_anomaly_score = models.FloatField(default=0.0)
    threshold_used = models.FloatField(default=0.0)
    overall_verdict = models.CharField(max_length=30, choices=VERDICT_CHOICES, default=VERDICT_BENIGN)
    latency_ms = models.FloatField(default=0.0)
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Scan #{self.id}: {self.file_name} -> {self.get_overall_verdict_display()}"

class DetectionResult(models.Model):
    ATTACK_NORMAL = 'NORMAL'
    ATTACK_PROBE = 'SUSPICIOUS_PROBE'
    ATTACK_INTRUSION = 'HIGH_RISK_INTRUSION'
    ATTACK_ZERO_DAY = 'ZERO_DAY_ANOMALY'

    ATTACK_CHOICES = [
        (ATTACK_NORMAL, 'Normal Baseline Traffic'),
        (ATTACK_PROBE, 'Suspicious Probe / Reconnaissance'),
        (ATTACK_INTRUSION, 'High Risk Intrusion Pattern'),
        (ATTACK_ZERO_DAY, 'Zero-Day Novel Anomaly Vector'),
    ]

    upload = models.ForeignKey(TrafficUpload, on_delete=models.CASCADE, related_name='results')
    chunk_id = models.PositiveIntegerField()
    start_idx = models.PositiveIntegerField(default=0)
    end_idx = models.PositiveIntegerField(default=0)
    flow_count = models.PositiveIntegerField(default=0)

    anomaly_score = models.FloatField()
    threshold = models.FloatField()
    is_malicious = models.BooleanField(default=False)
    attack_type = models.CharField(max_length=40, choices=ATTACK_CHOICES, default=ATTACK_NORMAL)
    reconstruction_error = models.FloatField(default=0.0)
    mahalanobis_dist = models.FloatField(default=0.0)

    class Meta:
        ordering = ['chunk_id']

    def __str__(self):
        return f"Upload #{self.upload.id} - Chunk #{self.chunk_id}: {self.attack_type} (Score={self.anomaly_score:.3f})"

class AlertRecord(models.Model):
    SEVERITY_LOW = 'LOW'
    SEVERITY_MEDIUM = 'MEDIUM'
    SEVERITY_HIGH = 'HIGH'
    SEVERITY_CRITICAL = 'CRITICAL'

    SEVERITY_CHOICES = [
        (SEVERITY_LOW, 'Low Severity'),
        (SEVERITY_MEDIUM, 'Medium Severity'),
        (SEVERITY_HIGH, 'High Severity'),
        (SEVERITY_CRITICAL, 'Critical Zero-Day Alert'),
    ]

    STATUS_NEW = 'NEW'
    STATUS_REVIEW = 'IN_REVIEW'
    STATUS_RESOLVED = 'RESOLVED'
    STATUS_FALSE_POS = 'FALSE_POSITIVE'

    STATUS_CHOICES = [
        (STATUS_NEW, 'New Alert'),
        (STATUS_REVIEW, 'Under Investigation'),
        (STATUS_RESOLVED, 'Incident Mitigated'),
        (STATUS_FALSE_POS, 'Verified False Positive'),
    ]

    upload = models.ForeignKey(TrafficUpload, on_delete=models.CASCADE, related_name='alerts')
    detection_result = models.ForeignKey(DetectionResult, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=250)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default=SEVERITY_HIGH)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_NEW)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_alerts'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.get_severity_display()}] {self.title} - Status: {self.get_status_display()}"
