"""
Master URL configuration for Zero-Day GraphIDS.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect

def root_redirect(request):
    """Redirect root path to dashboard or login."""
    if request.user.is_authenticated:
        return redirect('dashboard:index')
    return redirect('accounts:login')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', root_redirect, name='root'),
    path('accounts/', include('accounts.urls')),
    path('datasets/', include('datasets.urls')),
    path('training/', include('training.urls')),
    path('detection/', include('detection.urls')),
    path('dashboard/', include('dashboard.urls')),
    path('reports/', include('reports.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
