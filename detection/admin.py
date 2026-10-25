from django.contrib import admin
from .models import TrafficUpload, DetectionResult, AlertRecord

class DetectionResultInline(admin.TabularInline):
    model = DetectionResult
    extra = 0
    readonly_fields = ('chunk_id', 'anomaly_score', 'threshold', 'is_malicious', 'attack_type')

class AlertRecordInline(admin.TabularInline):
    model = AlertRecord
    extra = 0

@admin.register(TrafficUpload)
class TrafficUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file_name', 'user', 'model_used', 'overall_verdict', 'total_chunks', 'malicious_chunks', 'latency_ms', 'created_at')
    list_filter = ('overall_verdict', 'status', 'created_at')
    inlines = [DetectionResultInline, AlertRecordInline]

@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ('upload', 'chunk_id', 'anomaly_score', 'threshold', 'is_malicious', 'attack_type')
    list_filter = ('is_malicious', 'attack_type')

@admin.register(AlertRecord)
class AlertRecordAdmin(admin.ModelAdmin):
    list_display = ('title', 'severity', 'status', 'upload', 'created_at')
    list_filter = ('severity', 'status', 'created_at')
