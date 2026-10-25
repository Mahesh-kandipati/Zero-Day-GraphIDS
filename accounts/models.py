from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    """
    Custom user model supporting Role-Based Access Control (RBAC)
    for Zero-Day GraphIDS Security Operations Center (SOC).
    """
    ROLE_ADMIN = 'admin'
    ROLE_ANALYST = 'analyst'
    ROLE_USER = 'user'

    ROLE_CHOICES = [
        (ROLE_ADMIN, 'SOC Administrator'),
        (ROLE_ANALYST, 'Security Analyst'),
        (ROLE_USER, 'Standard User / Auditor'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default=ROLE_ANALYST,
        help_text="User authorization role within the SOC platform."
    )
    organization = models.CharField(
        max_length=150,
        blank=True,
        default='Default SOC Unit',
        help_text="Designated organizational unit or enterprise domain."
    )
    phone_number = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-date_joined']

    @property
    def is_soc_admin(self) -> bool:
        """Check if user has full administrative SOC permissions."""
        return self.is_superuser or self.role == self.ROLE_ADMIN

    @property
    def is_soc_analyst(self) -> bool:
        """Check if user has analyst capabilities."""
        return self.is_soc_admin or self.role == self.ROLE_ANALYST

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
