import torch
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer

class CodeSearchModel(pl.LightningModule):
    def __init__(self, model_name, lr=2e-5, batch_size=16):
        super().__init__()
        self.save_hyperparameters()
        self.model = SentenceTransformer(model_name)
        self.losses = []
    
    def encode_texts(self, texts):
        self.model[0].auto_model.train()
        encoded = self.model.tokenize(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        embeddings = self.model[0].auto_model(**encoded).last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def _compute_loss(self, batch, log_name):
        queries, codes = batch
        q_emb = self.encode_texts(queries)
        c_emb = self.encode_texts(codes)
        
        sim = torch.mm(q_emb, c_emb.t())
        labels = torch.arange(len(queries), device=sim.device)
        loss = torch.nn.functional.cross_entropy(sim, labels)
        
        self.log(log_name, loss, prog_bar=True, batch_size=len(queries))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, 'train_loss')
        self.losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'val_loss')
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

