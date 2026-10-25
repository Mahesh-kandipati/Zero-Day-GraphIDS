from django.contrib import admin
from .models import Dataset, GraphConfig, DatasetSplit

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name', 'dataset_type', 'status', 'num_records', 'num_features', 'uploaded_by', 'created_at')
    list_filter = ('dataset_type', 'status', 'created_at')
    search_fields = ('name', 'description')

@admin.register(GraphConfig)
class GraphConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'dataset', 'graph_type', 'k_neighbors', 'chunk_size', 'is_active')
    list_filter = ('graph_type', 'is_active')

@admin.register(DatasetSplit)
class DatasetSplitAdmin(admin.ModelAdmin):
    list_display = ('dataset', 'train_records', 'val_records', 'test_records', 'is_leakage_prevented')
