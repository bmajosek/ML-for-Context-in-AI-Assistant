import torch
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from config import OUTPUT_DIR, EVAL_K

class CodeSearchModel(pl.LightningModule):
    def __init__(self, model_name, lr=2e-5, batch_size=16, freeze_layers=True, loss_margin=0.3, loss_function='triplet'):
        super().__init__()
        self.save_hyperparameters()
        self.model = SentenceTransformer(model_name)
        if freeze_layers:
            self._freeze_all_but_last()
        self.losses = []
        self.val_losses = []
        self.train_epoch_losses = []
        self.val_epoch_losses = []

    def _freeze_all_but_last(self):
        encoder = self.model[0].auto_model
        enc_layers = list(encoder.encoder.layer)
        for layer in enc_layers[:-1]:
            for param in layer.parameters():
                param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * input_mask_expanded).sum(1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

    def encode_texts(self, texts):
        encoded = self.model.tokenize(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.set_grad_enabled(self.training):
            model_output = self.model[0].auto_model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def _compute_loss(self, batch, log_name):
        queries, codes = batch
        q_emb = self.encode_texts(queries)
        c_emb = self.encode_texts(codes)
        if self.hparams.loss_function == 'triplet':
            batch_losses = []
            margin = self.hparams.loss_margin
            for i in range(len(queries)):
                anchor = q_emb[i].unsqueeze(0)
                positive = c_emb[i].unsqueeze(0)
                sim_scores = torch.mm(anchor, c_emb.t())
                sim_scores[0, i] = -float('inf')
                hard_negative_idx = torch.argmax(sim_scores)
                negative = c_emb[hard_negative_idx].unsqueeze(0)
                ap_dist = 1 - torch.cosine_similarity(anchor, positive)
                an_dist = 1 - torch.cosine_similarity(anchor, negative)
                loss = torch.clamp(ap_dist - an_dist + margin, min=0.0)
                batch_losses.append(loss)
            final_loss = torch.mean(torch.stack(batch_losses))
        elif self.hparams.loss_function == 'cross_entropy':
            sim = torch.mm(q_emb, c_emb.t())
            labels = torch.arange(len(queries), device=sim.device)
            final_loss = torch.nn.functional.cross_entropy(sim, labels)
        else:
            raise ValueError(f"Unknown loss function: {self.hparams.loss_function}")
        self.log(log_name, final_loss, prog_bar=True, batch_size=len(queries))
        return final_loss
    
    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, 'train_loss')
        self.losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, 'val_loss')
        self.val_losses.append(loss.item())
        return loss
    
    def on_train_epoch_end(self):
        if self.losses:
            epoch_mean = np.mean(self.losses)
            self.train_epoch_losses.append(epoch_mean)
        self.losses.clear()

    def on_validation_epoch_end(self):
        if self.val_losses:
            epoch_mean = np.mean(self.val_losses)
            self.val_epoch_losses.append(epoch_mean)
        self.val_losses.clear()
        try:
            self.model.save(OUTPUT_DIR)
        except Exception:
            pass
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

