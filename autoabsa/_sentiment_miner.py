import torch
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments
)
from sentence_transformers import SentenceTransformer

class SentimentMiner:
    def __init__(self, 
                 pos_term="good", 
                 neg_term="bad",
                 neutral_term="ok", 
                 model_id="sentence-transformers/all-MiniLM-L12-v2") -> None:
        self.pos_term = pos_term
        self.neg_term = neg_term
        self.neutral_term = neutral_term
        self.model = SentenceTransformer(model_id)
        self.cosine_simlarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.softmax = torch.nn.Softmax(dim=0)

    def assign_polarity(self, opinion_word):
        pos_emb = torch.tensor(self.model.encode(self.pos_term)).unsqueeze(dim=0)
        neg_emb = torch.tensor(self.model.encode(self.pos_term)).unsqueeze(dim=0)
        neut_emb = torch.tensor(self.model.encode(self.pos_term)).unsqueeze(dim=0)
        sent_emb = torch.tensor(self.model.encode(opinion_word)).unsqueeze(dim=0)

        polarity = torch.argmax(
            self.softmax(torch.cat([
                self.cosine_simlarity(pos_emb, sent_emb),
                self.cosine_simlarity(neg_emb, sent_emb),
                self.cosine_simlarity(neut_emb, sent_emb)
                ])
            ))
        return polarity
