import pandas as pd
from autoabsa import (
    # DataHandler, DomainAdaptation,
    AspectTermMiner, OpinionWordMiner, SentimentMiner,
)
from typing import Union, Optional, Dict, List


class AutoABSA:
    def __init__(self) -> None:
        # self.da = DomainAdaptation()
        self.at = AspectTermMiner()
        self.owm = OpinionWordMiner()
        self.sm = SentimentMiner()

    # Step 1: Load Data
    # def load_data(self, docs):
    #     self.dataset = DataHandler(docs)

    def get_aspect_term(self, text):
        aspect_term = self.at.extract_aspect_term(text=text)
        return aspect_term

    def get_opinion_word(self, text, aspect):
        opinion_word, _ = self.owm.mine_opinion_words(text=text, aspect_word=aspect)
        return opinion_word

    def get_sentiment_polarity(self, text, aspect):
        opinion_word = self.get_opinion_word(text=text, aspect=aspect)
        opinion_word = self.sm.assign_polarity(opinion_words=opinion_word)
        return opinion_word

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
            docs: List[str]
            ):
        # Domain Adaptation Step
        pass

    def fit_transform(self,
                      docs,
                      extract_aspect: Optional[bool] = None,
                      extract_opinion: Optional[bool] = None,
                      extract_polarity: Optional[bool] = None
                      ):
        pass
