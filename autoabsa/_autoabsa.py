import spacy
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, PreTrainedModel
from datasets import DatasetDict, Dataset
from typing import Union, Optional, List
from autoabsa import DomainAdaptation, AspectTermMiner, OpinionWordMiner, SentimentMiner
from .utils import get_device


class AutoABSA:
    DEVICE = get_device()

    def __init__(self,
                 model_id: str = None,
                 tokenizer_name: str = None,
                 text_column: Optional[str] = None,
                 chunk_size: Optional[int] = None,
                 mlm_proba: Optional[float] = None,
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

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states=True, output_attentions=True)
        self.is_fitted = False

        self.domain_adaptation = DomainAdaptation(tokenizer=self.tokenizer,
                                                  model=self.model,
                                                  text_column='text' or text_column,
                                                  chunk_size=128 or chunk_size,
                                                  mlm_proba=0.15 or mlm_proba,
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

    def get_aspect_term(self, text: str) -> str:
        aspect_term = self.aspect_term_miner.extract_aspect_term(text=text)
        return aspect_term

    def get_opinion_word(self, text: str, aspect: str,
                         debug: Optional[bool] = False,
                         model: Optional[PreTrainedModel] = None
                         ) -> str:
        opinion_word, _ = self.opinion_word_miner.mine_opinion_words(text=text, aspect_word=aspect,
                                                                     debug=debug, model=model)
        return opinion_word

    def get_sentiment_polarity(self, opinion_words: Union[str, List[str]]) -> list:
        sentiment_polarities = self.sentiment_miner.assign_polarity(opinion_words=opinion_words)
        return sentiment_polarities

    def fit(self,
            docs: Union[List[str], Dataset, DatasetDict],
            test_docs: Optional[Union[List[str], Dataset]] = None,
            test_size: Optional[Union[int, float]] = None,
            return_trainer: Optional[bool] = False,
            **trainer_kwargs
            ) -> Union[Trainer, None]:
        trainer = self.domain_adaptation.pre_finetune(docs=docs,
                                                      test_docs=test_docs,
                                                      test_size=test_size,
                                                      **trainer_kwargs
                                                      )
        self.is_fitted = True
        if return_trainer:
            return trainer

    def transform(self,
                  docs: List[str],
                  extract_aspects: Optional[bool] = False,
                  aspect_term_list: Optional[List[str]] = None,
                  extract_opinion_words: Optional[bool] = False,
                  opinion_word_list: Optional[List[str]] = None,
                  debug: Optional[bool] = False,
                  fitted_model: Optional[PreTrainedModel] = None,
                  ) -> pd.DataFrame:

        if not self.is_fitted:
            print("Domain adaptation step not done. Using default pre-trained weights.")

        # Extract Aspect Terms
        if extract_aspects:
            aspect_term_list = []
            for text in docs:
                aspect_term_list.append(self.get_aspect_term(text=text))
        else:
            if aspect_term_list is None:
                raise Exception("aspect_term_list is required argument, when aspects are not extracted.")
            else:
                print('Using pre extracted aspect terms for opinion word mining.')

        assert len(aspect_term_list) == len(docs)

        # Extract Opinion Words
        if extract_opinion_words:
            opinion_word_list = []
            progress_bar = tqdm(zip(docs, aspect_term_list), desc="Extracting Opinion Words", total=len(docs))
            for text, aspect_term in progress_bar:
                opinion_word_list.append(self.get_opinion_word(text=text,
                                                               aspect=aspect_term,
                                                               debug=debug,
                                                               model=fitted_model))
                progress_bar.update(1)

        else:
            if opinion_word_list is None:
                raise Exception("opinion_word_list is required argument, when opinion words are not extracted.")

        assert len(opinion_word_list) == len(docs)

        # Assign Polarity
        polarity_list = self.get_sentiment_polarity(opinion_words=opinion_word_list)

        assert len(polarity_list) == len(docs)

        # Step 5: Return Opinion-Sentiment Tuples
        return_df = pd.DataFrame({'text': docs,
                                  'aspect_term': aspect_term_list,
                                  'opinion_word_pred': opinion_word_list,
                                  'polarity_pred': polarity_list
                                  })
        return_df['polarity_pred'] = return_df['polarity_pred'].map({0: 'POS', 1: 'NEG', 2: 'NEUT'})
        return return_df

    def fit_transform(self,
                      docs: List[str],
                      test_docs: Optional[Union[List[str], Dataset]] = None,
                      test_size: Optional[Union[int, float]] = None,
                      extract_aspects: Optional[bool] = False,
                      aspect_term_list: Optional[List[str]] = None,
                      extract_opinion_words: Optional[bool] = False,
                      opinion_word_list: Optional[List[str]] = None,
                      debug: Optional[bool] = False,
                      **trainer_kwargs
                      ):

        trainer = self.fit(docs, test_docs, test_size, return_trainer=True, **trainer_kwargs)
        print("Model fit completed. Starting transform.")
        ret_df = self.transform(docs, extract_aspects, aspect_term_list, extract_opinion_words,
                                opinion_word_list, debug, trainer.model)
        return ret_df
