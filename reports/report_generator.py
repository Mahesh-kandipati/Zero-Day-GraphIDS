"""
Forensic PDF and Data Export Generator for Zero-Day GraphIDS.
Produces SOC audit reports using ReportLab with cybersecurity styling.
"""

import os
from typing import Dict, Any
from django.conf import settings
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_detection_report(upload, output_path: str) -> str:
    """
    Builds a SOC Forensic PDF Report for an uninspected network traffic scan.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'SOCTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
        spaceAfter=15
    )
    section_style = ParagraphStyle(
        'SOCSection',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'SOCBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("<b>ZERO-DAY GRAPHIDS // FORENSIC SCAN REPORT</b>", title_style))
    elements.append(Paragraph(f"<b>Security Classification:</b> RESTRICTED // SOC INCIDENT AUDIT", body_style))
    elements.append(Spacer(1, 10))

    # Scan Summary Table
    summary_data = [
        ['Parameter', 'Audit Value'],
        ['Scan ID', f"#{upload.id}"],
        ['Traffic File', upload.file_name],
        ['GNN Model Used', upload.model_used.name],
        ['Overall Forensic Verdict', upload.get_overall_verdict_display()],
        ['Total Graph Chunks', str(upload.total_chunks)],
        ['Malicious / Novel Chunks', str(upload.malicious_chunks)],
        ['Benign Baseline Chunks', str(upload.normal_chunks)],
        ['Average Anomaly Score', f"{upload.avg_anomaly_score:.4f}"],
        ['Calibrated Threshold', f"{upload.threshold_used:.4f}"],
        ['Inference Latency', f"{upload.latency_ms:.2f} ms"],
        ['Timestamp', upload.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]

    t_summary = Table(summary_data, colWidths=[200, 300])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, -1), 4),
        ('TOPPADDING', (0, 0), (1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 15))

    # Chunk Verdicts Sample
    elements.append(Paragraph("<b>Detailed Chunk Anomaly Classifications</b>", section_style))

    results = upload.results.all()[:25]
    res_data = [['Chunk #', 'Score', 'Threshold', 'Verdict', 'Flows Range']]
    for r in results:
        res_data.append([
            f"Chunk {r.chunk_id}",
            f"{r.anomaly_score:.3f}",
            f"{r.threshold:.3f}",
            r.get_attack_type_display(),
            f"{r.start_idx} - {r.end_idx}"
        ])

    t_res = Table(res_data, colWidths=[70, 70, 70, 180, 110])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(t_res)

    doc.build(elements)
    return output_path

def generate_pdf_training_report(job, output_path: str) -> str:
    """
    Builds a Model Training & Zero-Day Evaluation Report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'SOCTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1,
        spaceAfter=15
    )
    body_style = ParagraphStyle(
        'SOCBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    elements = []
    elements.append(Paragraph("<b>ZERO-DAY GRAPHIDS // GNN TRAINING BENCHMARK</b>", title_style))
    elements.append(Spacer(1, 10))

    job_data = [
        ['Parameter', 'Training Metric'],
        ['Job Name', job.name],
        ['Baseline Dataset', job.dataset.name],
        ['Graph Configuration', job.graph_config.name],
        ['Epochs Optimized', str(job.epochs)],
        ['Learning Rate', str(job.learning_rate)],
        ['Embedding Dimension', str(job.embedding_dim)],
        ['Final Training Loss', f"{job.final_loss or 0.0:.4f}"],
        ['Zero-Day F1 Score', f"{job.f1_score or 0.0:.4f}"],
        ['Zero-Day ROC-AUC', f"{job.roc_auc or 0.0:.4f}"],
        ['Training Duration', f"{job.training_duration_sec:.2f} seconds"],
        ['Execution Date', job.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")],
    ]

    t_job = Table(job_data, colWidths=[200, 300])
    t_job.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, -1), 4),
        ('TOPPADDING', (0, 0), (1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
    ]))
    elements.append(t_job)

    doc.build(elements)
    return output_path
