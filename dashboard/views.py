import json
import torch
import numpy as np
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from datasets.models import Dataset, GraphConfig
from training.models import TrainingJob
from detection.models import TrafficUpload, DetectionResult, AlertRecord
from detection.alerts import get_soc_alert_metrics
from ml_engine.embeddings import compute_2d_projections
from datasets.graph_export import export_graph_to_cytoscape_json
from ml_engine.graph_builder import GraphBuilder

@login_required
def dashboard_home(request):
    """
    Main SOC Operations Center Dashboard.
    Visualizes live telemetry, detection verdicts, training progress, and alert queues.
    """
    datasets_count = Dataset.objects.count()
    total_flows = sum(d.num_records for d in Dataset.objects.all())
    models_count = TrainingJob.objects.filter(status=TrainingJob.STATUS_COMPLETED).count()
    scans_count = TrafficUpload.objects.count()

    # Alerts summary
    alert_metrics = get_soc_alert_metrics()

    # Recent scans & training jobs
    recent_scans = TrafficUpload.objects.all()[:5]
    recent_jobs = TrainingJob.objects.all()[:5]

    # Overall verdict distributions
    verdict_counts = {
        'Benign': TrafficUpload.objects.filter(overall_verdict=TrafficUpload.VERDICT_BENIGN).count(),
        'Suspicious': TrafficUpload.objects.filter(overall_verdict=TrafficUpload.VERDICT_SUSPICIOUS).count(),
        'High Risk': TrafficUpload.objects.filter(overall_verdict=TrafficUpload.VERDICT_HIGH_RISK).count(),
        'Zero-Day Alert': TrafficUpload.objects.filter(overall_verdict=TrafficUpload.VERDICT_ZERO_DAY).count(),
    }

    # Best performing model metrics
    best_job = TrainingJob.objects.filter(status=TrainingJob.STATUS_COMPLETED).order_by('-f1_score').first()

    context = {
        'datasets_count': datasets_count,
        'total_flows': total_flows,
        'models_count': models_count,
        'scans_count': scans_count,
        'alert_metrics': alert_metrics,
        'recent_scans': recent_scans,
        'recent_jobs': recent_jobs,
        'best_job': best_job,
        'verdict_counts_json': json.dumps(verdict_counts),
    }
    return render(request, 'dashboard/admin_dashboard.html', context)

@login_required
def embeddings_view(request):
    """
    2D Latent Representation Visualization of normal vs zero-day anomaly clusters.
    """
    latest_job = TrainingJob.objects.filter(status=TrainingJob.STATUS_COMPLETED).first()
    points = []

    # Generate demonstration embedding projection points
    if latest_job:
        # Generate synthetic cluster points reflecting normal vs anomalous regions
        np.random.seed(42)
        n_norm = 40
        n_anom = 10
        norm_x = np.random.normal(loc=0.0, scale=1.0, size=n_norm)
        norm_y = np.random.normal(loc=0.0, scale=1.0, size=n_norm)
        anom_x = np.random.normal(loc=3.5, scale=0.8, size=n_anom)
        anom_y = np.random.normal(loc=3.0, scale=0.8, size=n_anom)

        for i in range(n_norm):
            points.append({'x': round(float(norm_x[i]), 3), 'y': round(float(norm_y[i]), 3), 'label': 'Normal Baseline'})
        for i in range(n_anom):
            points.append({'x': round(float(anom_x[i]), 3), 'y': round(float(anom_y[i]), 3), 'label': 'Zero-Day Anomaly'})

    return render(request, 'dashboard/embeddings.html', {
        'latest_job': latest_job,
        'points_json': json.dumps(points)
    })

@login_required
def graph_explorer(request):
    """
    Interactive Network Topology Explorer:
    Renders k-NN metric similarity graphs and topological clustering statistics.
    """
    # Build demo network topology
    np.random.seed(42)
    demo_features = np.random.randn(25, 8).astype(np.float32)
    builder = GraphBuilder(k_neighbors=4)
    _, G = builder.build_knn_graph(demo_features)

    cytoscape_data = export_graph_to_cytoscape_json(G, max_nodes=50)

    return render(request, 'dashboard/graph_explorer.html', {
        'graph_json': json.dumps(cytoscape_data['elements']),
        'stats': cytoscape_data['stats']
    })

@login_required
def system_status(request):
    """System and environment diagnostics."""
    import sys, platform
    status_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device': 'CUDA (GPU)' if torch.cuda.is_available() else 'CPU (Windows Optimized)',
    }
    return render(request, 'dashboard/system_status.html', {'status_info': status_info})
