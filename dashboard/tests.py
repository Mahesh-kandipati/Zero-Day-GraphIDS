from django.test import TestCase, Client
from django.urls import reverse
from accounts.models import CustomUser

class DashboardViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = CustomUser.objects.create_user(
            username='soc_commander',
            password='CommanderPassword123!',
            role=CustomUser.ROLE_ADMIN
        )
        self.client.login(username='soc_commander', password='CommanderPassword123!')

    def test_dashboard_home_authenticated(self):
        response = self.client.get(reverse('dashboard:index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Zero-Day GraphIDS')

    def test_embeddings_view(self):
        response = self.client.get(reverse('dashboard:embeddings'))
        self.assertEqual(response.status_code, 200)

    def test_graph_explorer_view(self):
        response = self.client.get(reverse('dashboard:graph_explorer'))
        self.assertEqual(response.status_code, 200)

    def test_system_status_view(self):
        response = self.client.get(reverse('dashboard:system_status'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'PyTorch')
