import os
import tempfile
import numpy as np
import torch
from django.test import SimpleTestCase
from ml_engine.graph_builder import GraphBuilder
from ml_engine.feature_extractor import NetworkFeatureExtractor
from ml_engine.encoder import GraphSAGEEncoder
from ml_engine.self_supervised import SelfSupervisedGraphLoss
from ml_engine.embeddings import extract_chunk_embeddings, compute_2d_projections
from ml_engine.checkpoints import save_checkpoint, load_checkpoint

class MLEngineArchitectureTestCase(SimpleTestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.n_samples = 30
        self.n_features = 10
        self.features = np.random.randn(self.n_samples, self.n_features).astype(np.float32)

    def test_graph_builder_knn(self):
        builder = GraphBuilder(k_neighbors=4, metric='euclidean')
        pyg_data, nx_graph = builder.build_knn_graph(self.features)

        self.assertEqual(pyg_data.x.shape, (self.n_samples, self.n_features))
        self.assertEqual(pyg_data.edge_index.shape[0], 2)
        self.assertTrue(pyg_data.edge_index.shape[1] > 0)
        self.assertEqual(nx_graph.number_of_nodes(), self.n_samples)

    def test_encoder_forward_pass(self):
        builder = GraphBuilder(k_neighbors=3)
        pyg_data, _ = builder.build_knn_graph(self.features)

        encoder = GraphSAGEEncoder(in_dim=self.n_features, hidden_dim=16, embedding_dim=8)
        encoder.eval()

        with torch.no_grad():
            node_embeds, graph_embed, x_recon = encoder(pyg_data.x, pyg_data.edge_index)

        self.assertEqual(node_embeds.shape, (self.n_samples, 8))
        self.assertEqual(graph_embed.shape, (1, 16))  # mean + max pooling = 8 * 2
        self.assertEqual(x_recon.shape, (self.n_samples, self.n_features))

    def test_self_supervised_loss_and_backward(self):
        builder = GraphBuilder(k_neighbors=3)
        pyg_data, _ = builder.build_knn_graph(self.features)

        encoder = GraphSAGEEncoder(in_dim=self.n_features, hidden_dim=16, embedding_dim=8)
        criterion = SelfSupervisedGraphLoss(node_dim=8, graph_dim=16, contrastive_weight=0.5)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(criterion.parameters()), lr=0.01)

        encoder.train()
        optimizer.zero_grad()
        loss, metrics = criterion(encoder, pyg_data.x, pyg_data.edge_index)

        self.assertTrue(loss.item() > 0.0)
        self.assertIn('recon_loss', metrics)
        self.assertIn('contrastive_loss', metrics)

        loss.backward()
        optimizer.step()

    def test_checkpoint_save_and_load(self):
        encoder = GraphSAGEEncoder(in_dim=self.n_features, hidden_dim=16, embedding_dim=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "model_test.pt")
            save_checkpoint(
                file_path=ckpt_path,
                encoder=encoder,
                scaler_params={'mean_': [0.0] * self.n_features},
                threshold_info={'threshold': 0.85},
                metadata={'epochs': 5}
            )
            self.assertTrue(os.path.exists(ckpt_path))

            loaded = load_checkpoint(ckpt_path, device='cpu')
            self.assertEqual(loaded['model_config']['in_dim'], self.n_features)
            self.assertEqual(loaded['threshold_info']['threshold'], 0.85)
