from django.db import models
from django.conf import settings
from datasets.models import Dataset, GraphConfig

class TrainingJob(models.Model):
    STATUS_PENDING = 'PENDING'
    STATUS_RUNNING = 'RUNNING'
    STATUS_COMPLETED = 'COMPLETED'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Queued for Training'),
        (STATUS_RUNNING, 'Optimization in Progress'),
        (STATUS_COMPLETED, 'Training Completed'),
        (STATUS_FAILED, 'Training Run Failed'),
    ]

    name = models.CharField(max_length=200, help_text="Experimental run identifier.")
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='training_jobs')
    graph_config = models.ForeignKey(GraphConfig, on_delete=models.CASCADE, related_name='training_jobs')

    epochs = models.PositiveIntegerField(default=15)
    learning_rate = models.FloatField(default=0.005)
    hidden_dim = models.PositiveIntegerField(default=64)
    embedding_dim = models.PositiveIntegerField(default=32)
    batch_size = models.PositiveIntegerField(default=16)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    final_loss = models.FloatField(null=True, blank=True)
    training_duration_sec = models.FloatField(default=0.0)

    # Evaluation Metrics on Validation Set
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    pr_auc = models.FloatField(null=True, blank=True)

    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True, null=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='training_runs'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.get_status_display()}) - F1: {self.f1_score or 'N/A'}"

class TrainingMetric(models.Model):
    job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='metrics')
    epoch = models.PositiveIntegerField()
    train_loss = models.FloatField()
    recon_loss = models.FloatField()
    contrastive_loss = models.FloatField()
    val_loss = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['epoch']

    def __str__(self):
        return f"Job #{self.job.id} - Epoch {self.epoch}: Loss={self.train_loss:.4f}"

class EvaluationResult(models.Model):
    job = models.ForeignKey(TrainingJob, on_delete=models.CASCADE, related_name='evaluations')
    evaluation_name = models.CharField(max_length=200, default="Zero-Day Benchmark")
    protocol = models.CharField(max_length=100, default="Leave-One-Attack-Family-Out")
    excluded_attack_family = models.CharField(max_length=100, default="ZERO_DAY_UNKNOWN")

    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    roc_auc = models.FloatField(default=0.0)
    pr_auc = models.FloatField(default=0.0)
    false_positive_rate = models.FloatField(default=0.0)
    true_positive_rate = models.FloatField(default=0.0)

    confusion_matrix = models.JSONField(default=dict)
    baseline_comparisons = models.JSONField(default=dict)
    latency_ms_per_flow = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.evaluation_name} for Job #{self.job.id} - Zero-Day F1: {self.f1_score:.3f}"
