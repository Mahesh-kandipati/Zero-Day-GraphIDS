import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import TrainingJob, TrainingMetric, EvaluationResult
from datasets.models import Dataset, GraphConfig
from .tasks import start_training_job_async
from .trainer import run_self_supervised_training

@login_required
def training_list(request):
    """Displays historical training runs, loss trends, and artifact statuses."""
    jobs = TrainingJob.objects.all()
    return render(request, 'training/training_list.html', {'jobs': jobs})

@login_required
def training_create(request):
    """Form interface for launching new self-supervised graph neural training runs."""
    datasets = Dataset.objects.filter(status=Dataset.STATUS_READY)

    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        graph_config_id = request.POST.get('graph_config')
        name = request.POST.get('name', 'GNN Self-Supervised Run')
        epochs = int(request.POST.get('epochs', 15))
        learning_rate = float(request.POST.get('learning_rate', 0.005))
        hidden_dim = int(request.POST.get('hidden_dim', 64))
        embedding_dim = int(request.POST.get('embedding_dim', 32))

        dataset = get_object_or_404(Dataset, pk=dataset_id)
        graph_config = get_object_or_404(GraphConfig, pk=graph_config_id)

        job = TrainingJob.objects.create(
            name=name,
            dataset=dataset,
            graph_config=graph_config,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            created_by=request.user,
            status=TrainingJob.STATUS_PENDING
        )

        # Launch training (threaded or synchronous based on request or config)
        sync_mode = request.POST.get('sync_mode', 'false').lower() == 'true'
        if sync_mode:
            try:
                run_self_supervised_training(job.id)
                messages.success(request, f"Training run '{job.name}' completed successfully.")
            except Exception as e:
                messages.error(request, f"Training run encountered an error: {e}")
        else:
            start_training_job_async(job.id)
            messages.info(request, f"Training job '{job.name}' launched in background. Check telemetry below.")

        return redirect('training:detail', pk=job.pk)

    return render(request, 'training/training_create.html', {'datasets': datasets})

@login_required
def training_detail(request, pk):
    """Detailed telemetry, loss curve visualization, and checkpoint inspection."""
    job = get_object_or_404(TrainingJob, pk=pk)
    metrics = list(job.metrics.order_by('epoch').values('epoch', 'train_loss', 'recon_loss', 'contrastive_loss', 'val_loss'))
    evaluations = job.evaluations.all()

    epochs = [m['epoch'] for m in metrics]
    train_losses = [m['train_loss'] for m in metrics]
    recon_losses = [m['recon_loss'] for m in metrics]
    val_losses = [m['val_loss'] for m in metrics]

    context = {
        'job': job,
        'metrics': metrics,
        'evaluations': evaluations,
        'epochs_json': json.dumps(epochs),
        'train_losses_json': json.dumps(train_losses),
        'recon_losses_json': json.dumps(recon_losses),
        'val_losses_json': json.dumps(val_losses),
    }
    return render(request, 'training/training_detail.html', context)

@login_required
def training_retrain(request, pk):
    """Clones and re-initiates an existing training run."""
    old_job = get_object_or_404(TrainingJob, pk=pk)
    new_job = TrainingJob.objects.create(
        name=f"{old_job.name} (Retrain)",
        dataset=old_job.dataset,
        graph_config=old_job.graph_config,
        epochs=old_job.epochs,
        learning_rate=old_job.learning_rate,
        hidden_dim=old_job.hidden_dim,
        embedding_dim=old_job.embedding_dim,
        created_by=request.user,
        status=TrainingJob.STATUS_PENDING
    )
    start_training_job_async(new_job.id)
    messages.info(request, f"Retraining job #{new_job.id} dispatched.")
    return redirect('training:detail', pk=new_job.pk)
