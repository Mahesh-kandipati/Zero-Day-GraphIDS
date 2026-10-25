"""
Background Task Execution for Training Jobs.
Supports asynchronous thread execution for local development
and optional Celery task delegation for distributed production.
"""

import threading
import logging
from django.conf import settings
from .trainer import run_self_supervised_training

logger = logging.getLogger('training')

def start_training_job_async(job_id: int):
    """
    Launches training in a background thread to prevent blocking
    the Django HTTP web request on local setups without Celery.
    """
    if getattr(settings, 'USE_CELERY', False):
        try:
            from .celery_tasks import celery_train_job
            celery_train_job.delay(job_id)
            return
        except Exception as e:
            logger.warning(f"Celery dispatch failed: {e}. Falling back to threaded execution.")

    # Background thread execution
    thread = threading.Thread(
        target=run_self_supervised_training,
        args=(job_id,),
        daemon=True,
        name=f"TrainingJobWorker-{job_id}"
    )
    thread.start()
