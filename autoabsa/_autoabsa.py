import spacy
from sklearn.metrics.pairwise import rbf_kernel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import DatasetDict, Dataset
from typing import Union, Optional, List
from autoabsa import DomainAdaptation, AspectTermMiner, OpinionWordMiner, SentimentMiner
from .utils import get_device


class AutoABSA:
    DEVICE = get_device()

    def __init__(self,
                 model_id: str = None,
                 text_column: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 mlm_proba: Optional[float] = None,
                 return_da_trainer: Optional[bool] = None,
                 similarity_function: Optional = None,
                 gamma: Optional[float] = None,
                 we_layer_list: Optional[List[int]] = None,
                 score_by: Optional[Union[str]] = None,
                 spacy_model: Optional[str] = None,
                 pos_term: Optional[str] = None,
                 neg_term: Optional[str] = None,
                 neutral_term: Optional[str] = None,
                 sentiment_model_id: Optional[str] = None,
                 ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id).to(AutoABSA.DEVICE)
        self.is_fitted = False

        self.domain_adaptation = DomainAdaptation(tokenizer=self.tokenizer,
                                                  model=self.model,
                                                  text_column='text' or text_column,
                                                  chunk_size=128 or chunk_size,
                                                  mlm_proba=0.15 or mlm_proba,
                                                  return_trainer=False or return_da_trainer
                                                  )

        self.aspect_term_miner = AspectTermMiner()

        self.opinion_word_miner = OpinionWordMiner(
            tokenizer=self.tokenizer,
            model=self.model,
            similarity_function=rbf_kernel or similarity_function,
            we_layer_list=[0] if we_layer_list is None else we_layer_list,
            score_by='attentions' if (score_by == 'attention' or score_by is None) else 'hidden_states',
            spacy_model=spacy.load('en_core_web_lg') or spacy.load(spacy_model),
            gamma=0.0001 or gamma
        )

        self.sentiment_miner = SentimentMiner(pos_term="good" or pos_term,
                                              neg_term="bad" or neg_term,
                                              neutral_term="ok" or neutral_term,
                                              model_id="sentence-transformers/all-MiniLM-L12-v2" or sentiment_model_id)

    def get_aspect_term(self, text):
        aspect_term = self.aspect_term_miner.extract_aspect_term(text=text)
        return aspect_term

    def get_opinion_word(self, text, aspect):
        opinion_word, _ = self.opinion_word_miner.mine_opinion_words(text=text, aspect_word=aspect)
        return opinion_word

    def get_sentiment_polarity(self, text, aspect,
                               opinion_word: Optional = None
                               ):
        if opinion_word is None:
            opinion_word = self.get_opinion_word(text=text, aspect=aspect)
            sentiment_polarity = self.sentiment_miner.assign_polarity(opinion_words=opinion_word)
        else:
            sentiment_polarity = self.sentiment_miner.assign_polarity(opinion_words=opinion_word)
        return sentiment_polarity

    # def extract_tuples(self, docs):
    #
    #     # Step 2: Domain Adaptation
    #     self.da.pre_finetune()
    #
    #     # Step 3: Extract Candidate Terms including Multiword Terms
    #     opinion_word_list = []
    #     for sample in self.dataset:
    #         opinion_word_list.append(self.owm.mine_opinion_words())
    #
    #     # Step 4: Assign Polarity
    #     polarity_list = []
    #     for opinion_word in opinion_word_list:
    #         polarity_list.append(self.sm)
    #
    #     # Step 5: Return Opinion-Sentiment Tuples
    #     if len(self.dataset) > 1:
    #         ret_ = pd.DataFrame({'text': [],
    #                              'aspect_term': [],
    #                              'opinion_word': [],
    #                              'sentiment_polarity': []
    #                              })
    #     else:
    #         ret_ = {'text': [],
    #                 'aspect_term': [],
    #                 'opinion_word': [],
    #                 'sentiment_polarity': []
    #                 }
    #     return ret_

    def fit(self,
            docs: Union[List[str], Dataset, DatasetDict],
            test_docs: Optional[Union[List[str], Dataset]] = None,
            test_size: Optional[Union[int, float]] = None,
            **trainer_kwargs
            ):
        self.domain_adaptation.pre_finetune(docs=docs,
                                            test_docs=test_docs,
                                            test_size=test_size,
                                            **trainer_kwargs
                                            )

    def fit_transform(self,
                      docs,
                      extract_aspect: Optional[bool] = None,
                      extract_opinion: Optional[bool] = None,
                      extract_polarity: Optional[bool] = None
                      ):
        pass
