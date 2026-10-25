import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import TrafficUpload, DetectionResult, AlertRecord
from training.models import TrainingJob
from .detector import run_zero_day_detection
from .alerts import triage_alert

@login_required
def upload_traffic(request):
    """Form to submit uninspected network flow traffic for Zero-Day inspection."""
    models_available = TrainingJob.objects.filter(status=TrainingJob.STATUS_COMPLETED).exclude(model_file='')

    if request.method == 'POST':
        model_id = request.POST.get('model_used')
        uploaded_file = request.FILES.get('file')

        if not model_id or not uploaded_file:
            messages.error(request, "Please select a trained model and attach a valid traffic CSV.")
            return redirect('detection:upload')

        job = get_object_or_404(TrainingJob, pk=model_id)

        upload = TrafficUpload.objects.create(
            user=request.user,
            file=uploaded_file,
            file_name=uploaded_file.name,
            model_used=job,
            status=TrafficUpload.STATUS_PENDING
        )

        try:
            run_zero_day_detection(upload.id)
            messages.success(request, f"Detection complete! Scan #{upload.id} evaluated {upload.total_chunks} graph chunks.")
            return redirect('detection:result', pk=upload.pk)
        except Exception as e:
            messages.error(request, f"Inference pipeline encountered an error: {e}")
            return redirect('detection:history')

    return render(request, 'detection/upload_traffic.html', {'models_available': models_available})

@login_required
def detection_result(request, pk):
    """Displays detailed chunk-by-chunk anomaly scores, severity classification, and forensic alerts."""
    upload = get_object_or_404(TrafficUpload, pk=pk)
    results = upload.results.all()
    alerts = upload.alerts.all()

    chunk_ids = [r.chunk_id for r in results]
    scores = [r.anomaly_score for r in results]
    thresholds = [r.threshold for r in results]
    is_mal = [1 if r.is_malicious else 0 for r in results]

    context = {
        'upload': upload,
        'results': results,
        'alerts': alerts,
        'chunk_ids_json': json.dumps(chunk_ids),
        'scores_json': json.dumps(scores),
        'thresholds_json': json.dumps(thresholds),
        'is_mal_json': json.dumps(is_mal),
    }
    return render(request, 'detection/detection_result.html', context)

@login_required
def detection_history(request):
    """Chronological archive of uninspected traffic scans."""
    uploads = TrafficUpload.objects.all()
    return render(request, 'detection/detection_history.html', {'uploads': uploads})

@login_required
def alert_list(request):
    """SOC alert monitoring and incident triage dashboard."""
    alerts = AlertRecord.objects.all()
    status_filter = request.GET.get('status')
    severity_filter = request.GET.get('severity')

    if status_filter:
        alerts = alerts.filter(status=status_filter)
    if severity_filter:
        alerts = alerts.filter(severity=severity_filter)

    return render(request, 'detection/alert_list.html', {
        'alerts': alerts,
        'status_filter': status_filter,
        'severity_filter': severity_filter
    })

@login_required
def alert_update_status(request, pk):
    """Updates the status and triage notes for an incident alert."""
    alert = get_object_or_404(AlertRecord, pk=pk)
    if request.method == 'POST':
        new_status = request.POST.get('status', alert.status)
        notes = request.POST.get('notes', '')
        triage_alert(alert.id, new_status, request.user, notes)
        messages.success(request, f"Alert #{alert.id} status updated to {new_status}.")

    return redirect('detection:alerts')
