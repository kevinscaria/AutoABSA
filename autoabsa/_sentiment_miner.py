import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from typing import Optional, Union, List
from .utils import get_device
DEVICE = get_device()


class SentimentMiner:
    def __init__(self,
                 pos_term: Optional[str] = None,
                 neg_term: Optional[str] = None,
                 neutral_term: Optional[str] = None,
                 model_id: Optional[str] = None
                 ) -> None:
        self.pos_term = pos_term
        self.neg_term = neg_term
        self.neutral_term = neutral_term
        self.model = SentenceTransformer(model_id)

    def assign_polarity(self,
                        opinion_words: Union[str, List[str]],
                        batch_size: Optional[int] = 2048
                        ) -> list:
        """
        TODO: Write detailed function description.
        Can handle more than one opinion_words
        """
        pos_emb = torch.tensor(self.model.encode(self.pos_term)).unsqueeze(dim=0)
        neg_emb = torch.tensor(self.model.encode(self.neg_term)).unsqueeze(dim=0)
        neut_emb = torch.tensor(self.model.encode(self.neutral_term)).unsqueeze(dim=0)
        comparison_matrix = torch.cat([pos_emb, neg_emb, neut_emb], dim=0).unsqueeze(dim=1)

        if isinstance(opinion_words, list):
            polarity = []
            self.model = self.model.to(device=DEVICE)
            comparison_matrix = comparison_matrix.to(device=DEVICE)
            opinion_words = [opinion_words[i:i+batch_size] for i in range(0, len(opinion_words), batch_size)]
            for opinion_word_batch in tqdm(opinion_words, desc="Extracting Sentiment Polarity"):
                sent_emb = torch.tensor(self.model.encode(opinion_word_batch)).unsqueeze(dim=0).to(device=DEVICE)
                similarity_matrix = F.cosine_similarity(sent_emb, comparison_matrix, dim=-1).T
                polarity_batch = torch.argmax(F.softmax(similarity_matrix, dim=1), dim=1).detach().cpu().numpy().tolist()
                polarity.extend(polarity_batch)
        else:
            sent_emb = torch.tensor(self.model.encode(opinion_words)).unsqueeze(dim=0)
            similarity_matrix = F.cosine_similarity(sent_emb, comparison_matrix, dim=-1).T
            polarity = torch.argmax(F.softmax(similarity_matrix, dim=1), dim=1).detach().numpy().tolist()
        return polarity
