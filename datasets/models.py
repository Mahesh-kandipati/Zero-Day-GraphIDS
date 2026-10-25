from django.db import models
from django.conf import settings
from .validators import validate_dataset_file

class Dataset(models.Model):
    TYPE_CICIDS = 'CICIDS2017'
    TYPE_CTU13 = 'CTU13'
    TYPE_USTC = 'USTC_TFC2016'
    TYPE_SYNTHETIC = 'SYNTHETIC'
    TYPE_CUSTOM = 'CUSTOM'

    DATASET_TYPE_CHOICES = [
        (TYPE_CICIDS, 'CIC-IDS2017 Network Benchmark'),
        (TYPE_CTU13, 'CTU-13 Botnet Traffic'),
        (TYPE_USTC, 'USTC-TFC2016 Encrypted Malware'),
        (TYPE_SYNTHETIC, 'Synthetic Flow Demonstration'),
        (TYPE_CUSTOM, 'Custom Packet Capture Flow CSV'),
    ]

    STATUS_PENDING = 'PENDING'
    STATUS_PROCESSING = 'PROCESSING'
    STATUS_READY = 'READY'
    STATUS_FAILED = 'FAILED'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending Analysis'),
        (STATUS_PROCESSING, 'Ingesting & Preprocessing'),
        (STATUS_READY, 'Validated & Ready'),
        (STATUS_FAILED, 'Validation Failed'),
    ]

    name = models.CharField(max_length=200, help_text="Designation for this network dataset.")
    description = models.TextField(blank=True, help_text="Experimental notes and traffic origin.")
    dataset_type = models.CharField(max_length=50, choices=DATASET_TYPE_CHOICES, default=TYPE_CICIDS)
    file = models.FileField(upload_to='datasets/', validators=[validate_dataset_file])
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)

    num_records = models.PositiveIntegerField(default=0)
    num_features = models.PositiveIntegerField(default=0)
    label_column = models.CharField(max_length=100, blank=True, null=True)
    class_distribution = models.JSONField(default=dict, blank=True)
    feature_names = models.JSONField(default=list, blank=True)
    file_size_kb = models.FloatField(default=0.0)
    error_message = models.TextField(blank=True, null=True)

    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='uploaded_datasets'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.get_dataset_type_display()}) - {self.num_records} flows"

class GraphConfig(models.Model):
    GRAPH_KNN = 'KNN_SIMILARITY'
    GRAPH_HOST = 'HOST_FLOW_TOPOLOGY'

    GRAPH_TYPE_CHOICES = [
        (GRAPH_KNN, 'k-Nearest Neighbors (k-NN) Flow Metric Similarity'),
        (GRAPH_HOST, 'Host-to-Host Flow Communication Topology'),
    ]

    METRIC_EUCLIDEAN = 'euclidean'
    METRIC_COSINE = 'cosine'

    METRIC_CHOICES = [
        (METRIC_EUCLIDEAN, 'Euclidean Distance'),
        (METRIC_COSINE, 'Cosine Distance'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='graph_configs')
    name = models.CharField(max_length=150, default="Default Flow Graph Config")
    graph_type = models.CharField(max_length=40, choices=GRAPH_TYPE_CHOICES, default=GRAPH_KNN)
    k_neighbors = models.PositiveIntegerField(default=5, help_text="Number of nearest neighbors (k) for flow adjacency.")
    chunk_size = models.PositiveIntegerField(default=100, help_text="Temporal/batch flow grouping window size.")
    overlap_ratio = models.FloatField(default=0.0, help_text="Fractional sliding window overlap (0.0 - 0.5).")
    similarity_metric = models.CharField(max_length=20, choices=METRIC_CHOICES, default=METRIC_EUCLIDEAN)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} [k={self.k_neighbors}, chunk={self.chunk_size}] on {self.dataset.name}"

class DatasetSplit(models.Model):
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='split_info')
    train_ratio = models.FloatField(default=0.70)
    val_ratio = models.FloatField(default=0.15)
    test_ratio = models.FloatField(default=0.15)
    train_records = models.PositiveIntegerField(default=0)
    val_records = models.PositiveIntegerField(default=0)
    test_records = models.PositiveIntegerField(default=0)
    is_leakage_prevented = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Splits for {self.dataset.name}: {self.train_records}/{self.val_records}/{self.test_records}"
