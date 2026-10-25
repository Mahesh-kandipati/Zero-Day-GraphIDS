from django import forms
from .models import Dataset, GraphConfig

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ('name', 'description', 'dataset_type', 'file')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control bg-dark text-light border-secondary'})
            if name == 'description':
                field.widget.attrs.update({'rows': 3})

class GraphConfigForm(forms.ModelForm):
    class Meta:
        model = GraphConfig
        fields = ('name', 'graph_type', 'k_neighbors', 'chunk_size', 'overlap_ratio', 'similarity_metric')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            field.widget.attrs.update({'class': 'form-control bg-dark text-light border-secondary'})
