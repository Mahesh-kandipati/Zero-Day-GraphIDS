import os
import json
import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse, FileResponse, Http404
from django.conf import settings
from .models import Report
from training.models import TrainingJob
from detection.models import TrafficUpload
from .report_generator import generate_pdf_detection_report, generate_pdf_training_report

@login_required
def report_list(request):
    """Archive of all generated PDF audit and forensic reports."""
    reports = Report.objects.all()
    return render(request, 'reports/report_list.html', {'reports': reports})

@login_required
def generate_training_report_view(request, job_id):
    """Triggers generation of PDF training benchmark report."""
    job = get_object_or_404(TrainingJob, pk=job_id)
    reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
    filename = f"training_report_job_{job.id}.pdf"
    file_path = os.path.join(reports_dir, filename)

    generate_pdf_training_report(job, file_path)

    report = Report.objects.create(
        title=f"GNN Training Audit: {job.name}",
        report_type=Report.TYPE_TRAINING,
        file_format=Report.FORMAT_PDF,
        file=f"reports/{filename}",
        training_job=job,
        generated_by=request.user
    )

    messages.success(request, f"Forensic training report #{report.id} generated successfully.")
    return redirect('reports:list')

@login_required
def generate_detection_report_view(request, upload_id):
    """Triggers generation of PDF intrusion scan report."""
    upload = get_object_or_404(TrafficUpload, pk=upload_id)
    reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
    filename = f"detection_report_scan_{upload.id}.pdf"
    file_path = os.path.join(reports_dir, filename)

    generate_pdf_detection_report(upload, file_path)

    report = Report.objects.create(
        title=f"Intrusion Scan #{upload.id}: {upload.file_name}",
        report_type=Report.TYPE_DETECTION,
        file_format=Report.FORMAT_PDF,
        file=f"reports/{filename}",
        traffic_upload=upload,
        generated_by=request.user
    )

    messages.success(request, f"Forensic scan report #{report.id} compiled successfully.")
    return redirect('reports:list')

@login_required
def download_report(request, pk):
    """Securely downloads generated report file."""
    report = get_object_or_404(Report, pk=pk)
    if not report.file or not os.path.exists(report.file.path):
        raise Http404("Report file not found on disk.")
    return FileResponse(open(report.file.path, 'rb'), as_attachment=True, filename=os.path.basename(report.file.path))

@login_required
def export_scan_csv(request, upload_id):
    """Exports chunk-by-chunk detection results as CSV."""
    upload = get_object_or_404(TrafficUpload, pk=upload_id)
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="scan_{upload.id}_results.csv"'

    writer = csv.writer(response)
    writer.writerow(['Chunk_ID', 'Flow_Count', 'Start_Flow', 'End_Flow', 'Anomaly_Score', 'Threshold', 'Is_Malicious', 'Attack_Classification'])

    for r in upload.results.all():
        writer.writerow([
            r.chunk_id, r.flow_count, r.start_idx, r.end_idx,
            r.anomaly_score, r.threshold, r.is_malicious, r.attack_type
        ])

    return response

@login_required
def export_scan_json(request, upload_id):
    """Exports scan telemetry and classifications as JSON."""
    upload = get_object_or_404(TrafficUpload, pk=upload_id)
    data = {
        'scan_id': upload.id,
        'traffic_file': upload.file_name,
        'model_used': upload.model_used.name,
        'overall_verdict': upload.overall_verdict,
        'total_chunks': upload.total_chunks,
        'malicious_chunks': upload.malicious_chunks,
        'avg_anomaly_score': upload.avg_anomaly_score,
        'threshold': upload.threshold_used,
        'latency_ms': upload.latency_ms,
        'results': [
            {
                'chunk_id': r.chunk_id,
                'anomaly_score': r.anomaly_score,
                'threshold': r.threshold,
                'is_malicious': r.is_malicious,
                'attack_type': r.attack_type,
            }
            for r in upload.results.all()
        ]
    }
    return HttpResponse(json.dumps(data, indent=2), content_type='application/json')
