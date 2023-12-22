import re
import torch
import sklearn
import numpy as np
import pandas as pd
# from nltk import word_tokenize
# from sacremoses import MosesTokenizer
from functools import reduce
from collections import defaultdict
from typing import Union, List
from spacy.tokens import Doc
from spacy.language import Language
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedModel
from .utils import get_device

DEVICE = get_device()


class OpinionWordMiner:
    """
    The class is written for a batch_size 1 scenario
    TODO: Convert to a batch processing scenario 
    """

    # moses_tokenizer = MosesTokenizer()

    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 model: PreTrainedModel,
                 similarity_function: sklearn.metrics,
                 we_layer_list: List[int],
                 score_by: Union[str],
                 spacy_model: Language,
                 gamma: float
                 ):
        self.tokenizer = tokenizer
        self.model = model
        self.similarity_function = similarity_function
        self.we_layer_list = we_layer_list
        self.score_by = score_by
        self.nlp = spacy_model
        self.nlp.tokenizer = self.custom_tokenizer
        self.gamma = gamma

    def custom_tokenizer(self, text):
        # Implement custom regex tokenization logic
        tokens = [i.lower() for i in re.findall(r'\w+|\S', text)]
        return Doc(self.nlp.vocab, words=tokens)

    @staticmethod
    def _get_word_embeddings(model, layer_idx, score_by, tokenized_text):
        """
        This function extracts the word/token embeddings from the specified hidden layers.
        """

        model = model.to(DEVICE)
        tokenized_text = tokenized_text.to(DEVICE)

        tokenized_sentence_output = model(**tokenized_text, output_attentions=True)
        tokenized_sentence_hidden_states = torch.stack(tokenized_sentence_output[score_by], dim=0)
        tokenized_sentence_hidden_states = torch.squeeze(tokenized_sentence_hidden_states, dim=1)
        if score_by == 'attentions':
            # Extract the lower layer attention heads
            tokenized_sentence_embeddings = tokenized_sentence_hidden_states[2]  # (Num Heads, Seq Length, Seq Length)
        else:
            tokenized_sentence_embeddings = tokenized_sentence_hidden_states.permute(
                1, 0, 2)  # (Num Layers, Seq Length, Emb_Size)

        if score_by == 'attentions':
            # Extract specific heads for attention based scoring
            token_level_embeddings = tokenized_sentence_embeddings.mean(dim=0).detach().cpu().numpy()
        else:
            # Extract specific layer outputs for embedding based scoring
            token_vecs_cat = []
            for token in tokenized_sentence_embeddings:
                cat_vec = torch.mean(torch.cat(([token[i] for i in layer_idx]), dim=1), dim=1)
                token_vecs_cat.append(cat_vec.detach().numpy())
            token_level_embeddings = np.array(token_vecs_cat)
        return token_level_embeddings

    @staticmethod
    def _merge_subword_tokens(token_list):
        merged_tokens = []
        current_token = ""

        for token in token_list:
            if token.startswith("##"):
                current_token += token[2:]
            else:
                if current_token:
                    merged_tokens.append(current_token)
                    current_token = ""
                merged_tokens.append(token)

        # Check if there's a remaining token
        if current_token:
            merged_tokens.append(current_token)

        return merged_tokens

    @staticmethod
    def _get_aligned_subwords_embeddings(tokenizer, token_level_embeddings, score_by, text, debug=False):
        """
        Return word embeddings from hidden states when word piece/byte pair tokenizer is used.
        This method aligns the subwords into words by averaging the embeddings of subwords together
        """
        word_embeddings_aligned_list = []
        index_handler_for_cols = []
        tokens = tokenizer.tokenize(text)
        # manual_tokens = OpinionWordMiner._merge_subword_tokens(tokens)
        # nltk_tokens = word_tokenize(text)
        # moses_tokens = [i.lower() for i in OpinionWordMiner.moses_tokenizer.tokenize(text, escape=False)]
        regex_tokens = [i.lower() for i in re.findall(r'\w+|\S', text)]

        new_tokens = regex_tokens

        if debug:
            print("TOKENS: ", new_tokens)
            print("HF TOKENS: ", tokens)
            print("SHAPE: ", token_level_embeddings.shape)

        visited = defaultdict(int)
        for word in new_tokens:
            tokenized_token = tokenizer.tokenize(word)
            if len(tokenized_token) > 1:
                if tokenized_token[1] in visited:
                    second_idxs = [index for index, char in enumerate(tokens) if char == tokenized_token[1]]
                    second_idx = second_idxs[visited[tokenized_token[1]]]
                else:
                    second_idx = tokens.index(tokenized_token[1])
                start_idx = second_idx - 1
                visited[tokenized_token[1]] += 1
            else:
                if tokenized_token[0] in visited:
                    start_idxs = [index for index, char in enumerate(tokens) if char == tokenized_token[0]]
                    start_idx = start_idxs[visited[tokenized_token[0]]]
                else:
                    start_idx = tokens.index(tokenized_token[0])
            end_idx = start_idx + len(tokenized_token)
            word_embeddings = token_level_embeddings[start_idx:end_idx]
            if word_embeddings.shape[0] > 1:
                if debug:
                    print("WORDS: ", word)
                word_embeddings = np.mean(word_embeddings, axis=0).reshape(1, -1)
                index_handler_for_cols.append([start_idx, end_idx])

            word_embeddings_aligned_list.append(word_embeddings)

        word_embeddings_aligned = np.array(word_embeddings_aligned_list).squeeze(axis=1)

        if score_by == 'attentions':
            # Aligning Key Vectors
            diff_idx_ = 0
            for start_idx_, end_idx_ in index_handler_for_cols:
                if debug:
                    print("BEFORE: ", tokens[start_idx_:end_idx_], start_idx_, end_idx_, word_embeddings_aligned.shape)
                start_idx_ -= diff_idx_
                end_idx_ -= diff_idx_
                mean_val = np.mean(word_embeddings_aligned[:, start_idx_:end_idx_], axis=1, keepdims=True)
                word_embeddings_aligned[:, [start_idx_]] = mean_val
                word_embeddings_aligned = np.delete(word_embeddings_aligned, slice(start_idx_ + 1, end_idx_), axis=1)
                if debug:
                    print("AFTER: ", start_idx_, end_idx_, word_embeddings_aligned.shape)
                diff_idx_ += end_idx_ - start_idx_ - 1
                if debug:
                    print('DIFF: ', diff_idx_)

        if debug:
            print("NEW ALIGN SHAPE:", word_embeddings_aligned.shape)

        assert len(new_tokens) == word_embeddings_aligned.shape[0]

        if score_by == 'attentions':
            assert len(new_tokens) == word_embeddings_aligned.shape[1]

        return torch.tensor(word_embeddings_aligned), new_tokens

    @staticmethod
    def _filter_candidates(dataframe, shift_index_filter, pos_filter):
        """
        Method to create rules for extracting compound phrases.
        Arguments:
        dataframe: This argument is the dataframe with the POS and dependency information.
        index_filters: This argument handles the sequential indexes to check for the pos tags.
        pos_filters: This argument handles the pos tags at each index in the dependency dataframe.
        """
        filter_ = [(idx_, pos_) for idx_, pos_ in zip(shift_index_filter, pos_filter)]
        filter_condition = reduce(lambda x, y: x & y, [dataframe['pos'].shift(pos_) == val_ for pos_, val_ in filter_])

        compound_phrase_idx = []
        for idx in dataframe[filter_condition].index:
            if (idx + 1) < len(dataframe):
                comp_phrase_record = [
                    ' '.join([dataframe.loc[idx - idx_val]['opinion_word'] for idx_val in shift_index_filter]),
                    dataframe.loc[idx]['attention_score'],
                    '-'.join([dataframe.loc[idx - idx_val]['pos'] for idx_val in shift_index_filter]),
                    dataframe.loc[idx]['dep']
                ]
                for idx_val in shift_index_filter:
                    compound_phrase_idx.append(idx - idx_val)
                dataframe.loc[len(dataframe)] = comp_phrase_record
        dataframe.drop(index=compound_phrase_idx, inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def mine_opinion_words(self, text, aspect_word, debug=False, model=None):
        # Tokenize the text
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

        # Query the word embeddings/attention_weights for each token
        word_embeddings = self._get_word_embeddings(model=self.model if model is None else model,
                                                    layer_idx=self.we_layer_list,
                                                    score_by=self.score_by,
                                                    tokenized_text=tokenized_text
                                                    )

        # Align the word embeddings if the tokenizer splits words into sub words
        word_embeddings, aligned_tokens = self._get_aligned_subwords_embeddings(tokenizer=self.tokenizer,
                                                                                token_level_embeddings=word_embeddings,
                                                                                score_by=self.score_by,
                                                                                text=text,
                                                                                debug=debug
                                                                                )

        if debug:
            print("TEXT: ", text)
            print("ASPECT: ", aspect_word)

        # Extract POS tags and Dependency tags
        spacy_tokens_pos_tags = [token.pos_ for token in self.nlp(text)]
        spacy_tokens_deps = [token.dep_ for token in self.nlp(text)]

        # try:
        # Set default attention score for the aspect word
        aspect_word_score = [0 for i in range(len(aligned_tokens))]

        if self.score_by != 'attentions':
            # Only use similarity method if it is embedding based method
            try:
                self_attention_matrix = self.similarity_function(word_embeddings, word_embeddings, gamma=self.gamma)
            except:
                self_attention_matrix = self.similarity_function(word_embeddings, word_embeddings)
        else:
            # RBF Kernel is not required since attention weights already
            # show where the Query is attending to other Key vectors
            self_attention_matrix = word_embeddings

        if debug:
            print("AT: ", aligned_tokens)
        aspect_word = aspect_word.lower()

        # if ' ' in aspect_word or '-' in aspect_word:
        #     drop_records = []
        #     aspect_words = aspect_word.split()
        aspect_words = re.findall(r'\w+', aspect_word)

        if len(aspect_word) > 1:
            drop_records = []
            combined_aspect_word_embeddings = []
            for aspect_word in aspect_words:
                aspect_word_idx = aligned_tokens.index(aspect_word)
                drop_records.append(aspect_word_idx)
                combined_aspect_word_embeddings.append(self_attention_matrix[aspect_word_idx])
            aspect_word_score = np.mean(combined_aspect_word_embeddings, axis=0)
        else:
            aspect_word = aspect_words[0]
            aspect_word_idx = aligned_tokens.index(aspect_word)
            drop_records = [aspect_word_idx]
            aspect_word_score = self_attention_matrix[:, aspect_word_idx]

        dep_df = pd.DataFrame(aspect_word_score, columns=['attention_score'], index=aligned_tokens)
        dep_df['pos'] = spacy_tokens_pos_tags
        dep_df['dep'] = spacy_tokens_deps
        dep_df = dep_df.reset_index().rename(columns={'index': 'opinion_word'})
        dep_df.drop(index=drop_records, inplace=True)  # Remove aspect word scores (since it will be highest)

        # if debug:
        #     print(dep_df)

        # Reset index before applying rule based filtering
        dep_df.reset_index(drop=True, inplace=True)

        """
        TODO: Add option to generalize the rules 
        """

        # RULE 1: COMPOUND PHRASE EXTRACTION -> Compound Noun (Adjective + Noun) [AMOD]
        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1],
                                         pos_filter=['ADJ', 'NOUN'])

        # RULE 2: COMPOUND PHRASE EXTRACTION -> Compound Noun (Adjective + Noun + Noun) [AMOD~COMPOUND]
        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1, -2],
                                         pos_filter=['ADJ', 'NOUN', 'NOUN'])

        # RULE 3: COMPOUND PHRASE EXTRACTION -> Adverbial Phrase (Adverb + Adjective)
        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1],
                                         pos_filter=['ADV', 'ADJ'])

        if debug:
            print('KKKKK')
            print(dep_df)

        # RULE 4: COMPOUND PHRASE EXTRACTION -> Adverbial Phrase (ADP + NOUN)
        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1],
                                         pos_filter=['ADP', 'NOUN'])

        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1, -2],
                                         pos_filter=['VERB', 'VERB', 'ADV'])

        dep_df = self._filter_candidates(dataframe=dep_df,
                                         shift_index_filter=[0, -1],
                                         pos_filter=['VERB', 'ADV'])

        dep_df.reset_index(drop=True, inplace=True)

        if debug:
            print(dep_df)

        # Final filters
        dep_df = dep_df[(dep_df['pos'] == 'ADJ') |
                        (dep_df['pos'] == 'ADJ-NOUN') |
                        (dep_df['pos'] == 'ADJ-NOUN-NOUN') |
                        (dep_df['pos'] == 'ADV-ADJ') |
                        (dep_df['pos'] == 'ADP-NOUN') |
                        (dep_df['pos'] == 'VERB-VERB-ADV') |
                        (dep_df['pos'] == 'VERB-ADV')
                        ]

        if dep_df.shape[0] == 0:
            return 'NoRuleParsed', None

        # Candidate Reweighing
        opinion_word = dep_df.sort_values(by='attention_score', ascending=False).head(1)['opinion_word'].values[0]
        if debug:
            print(opinion_word)

        return opinion_word, None
        # except:
        #     return 'NoOpinionTerm', None
