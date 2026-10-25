from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    model = CustomUser
    list_display = ('username', 'email', 'role', 'organization', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('role', 'is_staff', 'is_active', 'organization')
    fieldsets = UserAdmin.fieldsets + (
        ('SOC Authorization & Organization', {
            'fields': ('role', 'organization', 'phone_number')
        }),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('SOC Authorization & Organization', {
            'fields': ('role', 'organization', 'phone_number')
        }),
    )
