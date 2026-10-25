"""
Alert Management and Incident Triage Engine for SOC Analysts.
"""

from typing import Dict, Any
from django.utils import timezone
from .models import AlertRecord

def triage_alert(alert_id: int, status: str, user, notes: str = "") -> AlertRecord:
    """
    Updates the operational state of an intrusion alert.
    """
    alert = AlertRecord.objects.get(pk=alert_id)
    alert.status = status
    if notes:
        alert.notes = f"{alert.notes}\n[{timezone.now().strftime('%Y-%m-%d %H:%M')}] {user.username}: {notes}".strip()
    if status == AlertRecord.STATUS_RESOLVED:
        alert.resolved_by = user
    alert.save()
    return alert

def get_soc_alert_metrics() -> Dict[str, Any]:
    """
    Aggregates active alert metrics across the SOC.
    """
    total = AlertRecord.objects.count()
    new_alerts = AlertRecord.objects.filter(status=AlertRecord.STATUS_NEW).count()
    in_review = AlertRecord.objects.filter(status=AlertRecord.STATUS_REVIEW).count()
    resolved = AlertRecord.objects.filter(status=AlertRecord.STATUS_RESOLVED).count()
    critical = AlertRecord.objects.filter(severity=AlertRecord.SEVERITY_CRITICAL).count()

    return {
        'total_alerts': total,
        'new_alerts': new_alerts,
        'in_review': in_review,
        'resolved': resolved,
        'critical': critical
    }
