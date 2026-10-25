from django.contrib import admin
from .models import TrainingJob, TrainingMetric, EvaluationResult

class TrainingMetricInline(admin.TabularInline):
    model = TrainingMetric
    extra = 0
    readonly_fields = ('epoch', 'train_loss', 'recon_loss', 'contrastive_loss', 'val_loss', 'timestamp')

@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ('name', 'dataset', 'status', 'epochs', 'final_loss', 'f1_score', 'roc_auc', 'created_at')
    list_filter = ('status', 'created_at')
    inlines = [TrainingMetricInline]

@admin.register(EvaluationResult)
class EvaluationResultAdmin(admin.ModelAdmin):
    list_display = ('job', 'evaluation_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc')
