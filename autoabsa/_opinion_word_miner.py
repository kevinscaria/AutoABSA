import re
import torch
import sklearn
import numpy as np
import pandas as pd
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
        self.tokenizer_type = None
        self.model = model
        self.similarity_function = similarity_function
        self.we_layer_list = we_layer_list
        self.score_by = score_by
        self.nlp = spacy_model
        self.nlp.tokenizer = self.custom_tokenizer
        self.gamma = gamma

    def custom_tokenizer(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids, tokens = OpinionWordMiner._get_aligned_output_tokens(self.tokenizer_type, tokens)
        tokens = [''.join(tok).replace("#", "").
                  replace("Ġ", "").
                  replace("▁", "") for tok in tokens]
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
                cat_vec = torch.mean(torch.cat(([token[i] for i in layer_idx]), dim=1), dim=1).tolist()
                token_vecs_cat.append(cat_vec.detach())
            token_level_embeddings = np.array(token_vecs_cat)
        return token_level_embeddings

    @staticmethod
    def _get_aligned_output_tokens(tokenizer_type, input_tokens):
        output_ids = []
        output_tokens = []
        current_idx = 0

        if tokenizer_type == 'SentencePiece':
            for token in input_tokens:
                if token.startswith('▁'):
                    output_ids.append([current_idx])
                    output_tokens.append([token])
                    current_idx += 1
                else:
                    output_ids[-1].append(current_idx)
                    output_tokens[-1].append(token)
                    current_idx += 1
            return output_ids, output_tokens

        elif tokenizer_type == "WordPiece":
            for token in input_tokens:
                if token.startswith("##"):
                    output_ids[-1].append(current_idx)
                    output_tokens[-1].append(token)
                    current_idx += 1
                else:
                    output_ids.append([current_idx])
                    output_tokens.append([token])
                    current_idx += 1
            return output_ids, output_tokens

        elif tokenizer_type == "BPE":
            output_tokens = [[input_tokens[0]]]
            output_ids = [[0]]
            for token in input_tokens[1:]:
                if token.startswith('Ġ'):
                    output_ids.append([current_idx])
                    output_tokens.append([token])
                    current_idx += 1
                else:
                    output_ids[-1].append(current_idx)
                    output_tokens[-1].append(token)
                    current_idx += 1
            return output_ids, output_tokens

        else:
            return None, None

    @staticmethod
    def _get_aligned_subwords_embeddings(token_level_embeddings, score_by, aligned_tokens,
                                         aligned_token_ids, debug=False):
        """
        Return word embeddings from hidden states when word piece/byte pair tokenizer is used.
        This method aligns the subwords into words by averaging the embeddings of subwords together
        """
        word_embeddings_aligned_list = []
        index_handler_for_cols = []

        if debug:
            print("Initial Word Embeddings Shape: ", token_level_embeddings.shape)

        for output_token_list in aligned_token_ids:
            if len(output_token_list) > 1:
                start_idx = output_token_list[0]
                end_idx = output_token_list[-1] + 1
            else:
                start_idx = output_token_list[0]
                end_idx = start_idx + 1

            word_embeddings = token_level_embeddings[start_idx:end_idx]

            if word_embeddings.shape[0] > 1:
                word_embeddings = np.mean(word_embeddings, axis=0).reshape(1, -1)
                index_handler_for_cols.append([start_idx, end_idx])

            word_embeddings_aligned_list.append(word_embeddings)

        word_embeddings_aligned = np.array(word_embeddings_aligned_list).squeeze(axis=1)

        if debug:
            print("Query Aligned Word Embeddings Shape: ", word_embeddings_aligned.shape)

        if score_by == 'attentions':
            # Aligning Key Vectors
            diff_idx_ = 0
            for start_idx_, end_idx_ in index_handler_for_cols:
                start_idx_ -= diff_idx_
                end_idx_ -= diff_idx_
                mean_val = np.mean(word_embeddings_aligned[:, start_idx_:end_idx_], axis=1, keepdims=True)
                word_embeddings_aligned[:, [start_idx_]] = mean_val
                word_embeddings_aligned = np.delete(word_embeddings_aligned, slice(start_idx_ + 1, end_idx_), axis=1)
                diff_idx_ += end_idx_ - start_idx_ - 1

        if debug:
            print("Key Aligned Word Embeddings Shape:", word_embeddings_aligned.shape)

        assert len(aligned_tokens) == word_embeddings_aligned.shape[0]

        if score_by == 'attentions':
            assert len(aligned_tokens) == word_embeddings_aligned.shape[1]

        return word_embeddings_aligned

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
                    # dataframe.loc[idx]['attention_score'],
                    sum([dataframe.loc[idx - idx_val]['attention_score'] for idx_val in shift_index_filter]),
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

        # Get tokenized tokens as tensors with input_ids
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

        # Query the word embeddings/attention_weights for each token
        word_embeddings = self._get_word_embeddings(model=self.model if model is None else model,
                                                    layer_idx=self.we_layer_list,
                                                    score_by=self.score_by,
                                                    tokenized_text=tokenized_text
                                                    )

        # Get raw tokenized tokens as string
        tokens = self.tokenizer.tokenize(text)

        # Understand Tokenizer Type
        if len(tokens) > 1 and tokens[1].startswith("Ġ"):
            self.tokenizer_type = "BPE"
        elif tokens[0].startswith("▁"):
            self.tokenizer_type = "SentencePiece"
        else:
            self.tokenizer_type = "WordPiece"

        # Get aligned tokenized tokens as string
        aligned_token_ids, aligned_tokens = (OpinionWordMiner.
                                             _get_aligned_output_tokens(tokenizer_type=self.tokenizer_type,
                                                                        input_tokens=tokens))

        aligned_tokens = [''.join(tok).replace("#", "").
                          replace("Ġ", "").
                          replace("▁", "").lower() for tok in aligned_tokens]

        if aligned_token_ids is None:
            raise Exception(
                "Unknown tokenizer. Aligning subword tokens may lead to error. "
                "Use one of SentencePiece, WordPiece, or BPE."
            )

        if debug:
            print("Text: ", text)
            print("Tokenizer Type: ", self.tokenizer_type)
            print("Tokenizer Output: ", tokens)
            print('Aligned Tokens: ', aligned_tokens)

        aligned_word_embeddings = self._get_aligned_subwords_embeddings(token_level_embeddings=word_embeddings,
                                                                        score_by=self.score_by,
                                                                        aligned_tokens=aligned_tokens,
                                                                        aligned_token_ids=aligned_token_ids,
                                                                        debug=debug
                                                                        )

        # Extract POS tags and Dependency tags
        spacy_tokens_pos_tags = [token.pos_ for token in self.nlp(text)]
        spacy_tokens_deps = [token.dep_ for token in self.nlp(text)]

        if debug:
            print("Spacy Tokenizer: ", [token.text.lower() for token in self.nlp(text)])
            print("Spacy Tokens Length: ", len(spacy_tokens_deps))

        # try:

        # Set default attention score for the aspect word
        aspect_word_score = [0 for i in range(len(tokens))]

        if self.score_by != 'attentions':
            # Only use similarity method if it is embedding based method
            try:
                self_attention_matrix = self.similarity_function(aligned_word_embeddings, aligned_word_embeddings,
                                                                 gamma=self.gamma)
            except:
                self_attention_matrix = self.similarity_function(aligned_word_embeddings, aligned_word_embeddings)
        else:
            # RBF Kernel is not required since attention weights already
            # show where the Query is attending to other Key vectors
            self_attention_matrix = aligned_word_embeddings
            if debug:
                print("Attention Matrix Shape: ", self_attention_matrix.shape)

        # try:
        aspect_word = aspect_word.strip()
        index_of_aspect_term_in_sentence = text.find(aspect_word)
        if index_of_aspect_term_in_sentence >= 0:
            if index_of_aspect_term_in_sentence > 0 and not text[index_of_aspect_term_in_sentence - 1].isalnum():
                aspect_word = " " + aspect_word

            aspect_words = [i.text.lower() for i in self.nlp(aspect_word)]

            if debug:
                print("Original Aspect Term: ", aspect_word)
                print("Tokenized Aspect Term: ", aspect_words)

            if len(aspect_word) > 1:
                drop_records = []
                combined_aspect_word_embeddings = []
                for aspect_subword in aspect_words:
                    aspect_word_idx = aligned_tokens.index(aspect_subword)
                    drop_records.append(aspect_word_idx)
                    combined_aspect_word_embeddings.append(self_attention_matrix[aspect_word_idx])
                aspect_word_score = np.mean(combined_aspect_word_embeddings, axis=0)
            else:
                aspect_word = aspect_words[0]
                aspect_word_idx = aligned_tokens.index(aspect_word)
                drop_records = [aspect_word_idx]
                aspect_word_score = self_attention_matrix[:, aspect_word_idx]

            if debug:
                print("Aspect Term Index: ", drop_records)
                if len(drop_records) > 1:
                    lower_idx_ = drop_records[0]
                    upper_idx_ = drop_records[-1]
                else:
                    lower_idx_ = drop_records[0]
                    upper_idx_ = lower_idx_ + 1
                print("Check Aspect Term Index: ", aligned_tokens[lower_idx_:upper_idx_])

            dep_df = pd.DataFrame(aspect_word_score, columns=['attention_score'], index=aligned_tokens)
            dep_df['pos'] = spacy_tokens_pos_tags
            dep_df['dep'] = spacy_tokens_deps
            dep_df = dep_df.reset_index().rename(columns={'index': 'opinion_word'})

            # Heuristic scaling by distance
            dep_df['distance'] = abs(dep_df.index - drop_records[0])
            dep_df['distance_influence'] = dep_df['distance'].apply(lambda x: 1 / np.exp(x))

            if debug:
                print("Computing Influence of Distance")
                print(dep_df)

            dep_df['attention_score'] = dep_df['attention_score'] + dep_df['distance_influence']
            dep_df.drop(columns=['distance', 'distance_influence'], axis=1, inplace=True)
            dep_df.drop(index=drop_records, inplace=True)  # Remove aspect words scores (since it will be highest)

            if debug:
                print("Before Rule Filters")
                print(dep_df)

            # Reset index before applying rule based filtering
            dep_df.reset_index(drop=True, inplace=True)

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

            # RULE 4: COMPOUND PHRASE EXTRACTION -> Adverbial Phrase (ADP + NOUN)
            dep_df = self._filter_candidates(dataframe=dep_df,
                                             shift_index_filter=[0, -1],
                                             pos_filter=['ADP', 'NOUN'])

            dep_df = self._filter_candidates(dataframe=dep_df,
                                             shift_index_filter=[0, -1],
                                             pos_filter=['ADV', 'ADV', 'VERB'])

            dep_df = self._filter_candidates(dataframe=dep_df,
                                             shift_index_filter=[0, -1],
                                             pos_filter=['ADV', 'VERB'])

            dep_df = self._filter_candidates(dataframe=dep_df,
                                             shift_index_filter=[0, -1],
                                             pos_filter=['VERB', 'VERB', 'ADV'])

            dep_df = self._filter_candidates(dataframe=dep_df,
                                             shift_index_filter=[0, -1],
                                             pos_filter=['VERB', 'ADV'])

            dep_df.reset_index(drop=True, inplace=True)

            # Final filters
            dep_df = dep_df[(dep_df['pos'] == 'ADJ') |
                            (dep_df['pos'] == 'ADJ-NOUN') |
                            (dep_df['pos'] == 'ADJ-NOUN-NOUN') |
                            (dep_df['pos'] == 'ADV-ADJ') |
                            (dep_df['pos'] == 'ADP-NOUN') |
                            (dep_df['pos'] == 'ADV-ADV-VERB') |
                            (dep_df['pos'] == 'ADV-VERB') |
                            (dep_df['pos'] == 'VERB-VERB-ADV') |
                            (dep_df['pos'] == 'VERB-ADV')
                            ]

            if debug:
                print("After Rule Filters")
                print(dep_df)

            if dep_df.shape[0] == 0:
                return 'NoRuleParsed', None

            # Candidate Reweighing
            opinion_word = dep_df.sort_values(by='attention_score', ascending=False).head(1)['opinion_word'].values[0]
            if debug:
                print("Aspect Term: ", ' '.join(aspect_words))
                print("Extracted Opinion Word: ", opinion_word)
            return opinion_word, None
            # except:
            #         #     return "Error", None
        else:
            print("Aspect term not present")
            opinion_word = "AspectTermNotFound"
            print("Extracted Opinion Word: ", opinion_word)
            return opinion_word, None
