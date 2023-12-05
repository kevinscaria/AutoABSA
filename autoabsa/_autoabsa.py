import pandas as pd
from autoabsa import (
    DataHandler, DomainAdaptation, 
    OpinionWordMiner, SentimentMiner, 
)

class AutoABSA:
    def __init__(self) -> None:
        self.da = DomainAdaptation()
        self.owm = OpinionWordMiner()
        self.sm = SentimentMiner()
    
    # Step 1: Load Data
    def load_data(self, docs):
        self.dataset = DataHandler(docs)


    def extract_tuples(self, docs):

        # Step 2: Domain Adaptation
        self.da.pre_finetune()

        # Step 3: Extract Candidate Terms including Multiword Terms
        opinion_word_list = []
        for sample in self.dataset:
            opinion_word_list.append(self.owm.mine_opinion_words())

        # Step 4: Assign Polarity
        polarity_list = []
        for opinion_word in opinion_word_list:
            polarity_list.append(self.sm)

        # Step 5: Return Opinion-Sentiment Tuples
        if len(self.dataset)>1:
            ret_ = pd.DataFrame({'text':[],
                                'aspect_term':[],
                                'opinion_word':[],
                                'sentiment_polarity':[]
                                })
        else:
            ret_ = {'text':[],
                    'aspect_term':[],
                    'opinion_word':[],
                    'sentiment_polarity':[]
                    }
        return ret_