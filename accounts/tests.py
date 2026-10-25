from django.test import TestCase, Client
from django.urls import reverse
from .models import CustomUser

class AccountsModelAndViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = CustomUser.objects.create_user(
            username='testanalyst',
            email='analyst@soc.local',
            password='Password123!',
            role=CustomUser.ROLE_ANALYST,
            organization='CyberDefense Corp'
        )

    def test_custom_user_creation_and_roles(self):
        self.assertEqual(self.user.username, 'testanalyst')
        self.assertTrue(self.user.is_soc_analyst)
        self.assertFalse(self.user.is_soc_admin)
        self.assertEqual(self.user.organization, 'CyberDefense Corp')

    def test_login_flow(self):
        response = self.client.post(reverse('accounts:login'), {
            'username': 'testanalyst',
            'password': 'Password123!'
        })
        self.assertEqual(response.status_code, 302)

    def test_profile_requires_login(self):
        response = self.client.get(reverse('accounts:profile'))
        self.assertEqual(response.status_code, 302)

    def test_profile_authenticated(self):
        self.client.login(username='testanalyst', password='Password123!')
        response = self.client.get(reverse('accounts:profile'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'CyberDefense Corp')
