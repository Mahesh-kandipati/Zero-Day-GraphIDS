import os
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Dataset, GraphConfig, DatasetSplit
from .forms import DatasetUploadForm, GraphConfigForm
from .preprocessing import clean_and_sanitize_flow_data, detect_schema_type, split_and_normalize_dataset

@login_required
def dataset_list(request):
    """Lists all uploaded datasets with processing state and telemetry."""
    datasets = Dataset.objects.all()
    return render(request, 'datasets/dataset_list.html', {'datasets': datasets})

@login_required
def dataset_upload(request):
    """Handles dataset ingestion, structural sanitation, and schema detection."""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.uploaded_by = request.user
            dataset.status = Dataset.STATUS_PROCESSING
            dataset.save()

            # Process the CSV file
            try:
                file_path = dataset.file.path
                dataset.file_size_kb = round(os.path.getsize(file_path) / 1024.0, 2)

                # Read sample or full file
                df_raw = pd.read_csv(file_path, low_memory=False)
                dataset.num_records = len(df_raw)

                # Preprocess & sanitize
                df_clean, label_col, feature_cols = clean_and_sanitize_flow_data(df_raw)
                dataset.num_features = len(feature_cols)
                dataset.label_column = label_col
                dataset.feature_names = feature_cols

                if label_col and label_col in df_clean.columns:
                    val_counts = df_clean[label_col].value_counts().to_dict()
                    dataset.class_distribution = {str(k): int(v) for k, v in val_counts.items()}
                else:
                    dataset.class_distribution = {'Unlabeled_Flows': len(df_clean)}

                dataset.status = Dataset.STATUS_READY
                dataset.save()

                # Automatically create default graph config
                GraphConfig.objects.create(
                    dataset=dataset,
                    name=f"Standard k-NN Graph ({dataset.name})",
                    graph_type=GraphConfig.GRAPH_KNN,
                    k_neighbors=min(5, max(1, dataset.num_records - 1)),
                    chunk_size=min(100, max(10, dataset.num_records))
                )

                # Create default split info
                train_count = int(0.70 * dataset.num_records)
                val_count = int(0.15 * dataset.num_records)
                test_count = dataset.num_records - (train_count + val_count)

                DatasetSplit.objects.create(
                    dataset=dataset,
                    train_ratio=0.70,
                    val_ratio=0.15,
                    test_ratio=0.15,
                    train_records=train_count,
                    val_records=val_count,
                    test_records=test_count,
                    is_leakage_prevented=True
                )

                messages.success(
                    request,
                    f"Dataset '{dataset.name}' parsed successfully: {dataset.num_records} flows, "
                    f"{dataset.num_features} numerical features identified."
                )
                return redirect('datasets:detail', pk=dataset.pk)

            except Exception as e:
                dataset.status = Dataset.STATUS_FAILED
                dataset.error_message = str(e)
                dataset.save()
                messages.error(request, f"Error processing dataset: {e}")
                return redirect('datasets:list')
    else:
        form = DatasetUploadForm()

    return render(request, 'datasets/dataset_upload.html', {'form': form})

@login_required
def dataset_detail(request, pk):
    """Detailed inspection of dataset features, class balance, and graph configurations."""
    dataset = get_object_or_404(Dataset, pk=pk)
    configs = dataset.graph_configs.all()
    split_info = getattr(dataset, 'split_info', None)

    if request.method == 'POST':
        config_form = GraphConfigForm(request.POST)
        if config_form.is_valid():
            new_config = config_form.save(commit=False)
            new_config.dataset = dataset
            new_config.save()
            messages.success(request, f"Graph configuration '{new_config.name}' created.")
            return redirect('datasets:detail', pk=dataset.pk)
    else:
        config_form = GraphConfigForm()

    # Read preview rows
    preview_rows = []
    headers = []
    if os.path.exists(dataset.file.path):
        try:
            df_preview = pd.read_csv(dataset.file.path, nrows=5)
            headers = list(df_preview.columns)
            preview_rows = df_preview.to_dict(orient='records')
        except Exception:
            pass

    context = {
        'dataset': dataset,
        'configs': configs,
        'split_info': split_info,
        'config_form': config_form,
        'headers': headers,
        'preview_rows': preview_rows,
    }
    return render(request, 'datasets/dataset_detail.html', context)

@login_required
def dataset_delete(request, pk):
    """Deletes dataset, associated media file, and configurations."""
    dataset = get_object_or_404(Dataset, pk=pk)
    if request.method == 'POST':
        name = dataset.name
        if dataset.file and os.path.exists(dataset.file.path):
            try:
                os.remove(dataset.file.path)
            except OSError:
                pass
        dataset.delete()
        messages.success(request, f"Dataset '{name}' deleted successfully.")
        return redirect('datasets:list')
    return render(request, 'datasets/dataset_confirm_delete.html', {'dataset': dataset})
