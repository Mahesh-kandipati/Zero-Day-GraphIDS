from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import CustomUserCreationForm, CustomAuthenticationForm, UserProfileForm
from .models import CustomUser

def login_view(request):
    """Handles user authentication with SOC session establishment."""
    if request.user.is_authenticated:
        return redirect('dashboard:index')

    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}! Access level: {user.get_role_display()}.")
            next_url = request.GET.get('next')
            if next_url:
                return redirect(next_url)
            return redirect('dashboard:index')
        else:
            messages.error(request, "Invalid username or password credentials.")
    else:
        form = CustomAuthenticationForm()

    return render(request, 'accounts/login.html', {'form': form})

def register_view(request):
    """Registers a new user with standard credentials and role assignment."""
    if request.user.is_authenticated:
        return redirect('dashboard:index')

    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f"Account successfully provisioned for {user.username}.")
            return redirect('dashboard:index')
        else:
            messages.error(request, "Registration validation error. Please review the highlighted fields.")
    else:
        form = CustomUserCreationForm()

    return render(request, 'accounts/register.html', {'form': form})

def logout_view(request):
    """Logs out user and invalidates session."""
    logout(request)
    messages.info(request, "You have been securely signed out of Zero-Day GraphIDS.")
    return redirect('accounts:login')

@login_required
def profile_view(request):
    """User profile and SOC security clearance management."""
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Security profile updated successfully.")
            return redirect('accounts:profile')
        else:
            messages.error(request, "Unable to save profile modifications.")
    else:
        form = UserProfileForm(instance=request.user)

    return render(request, 'accounts/profile.html', {'form': form, 'user_obj': request.user})
