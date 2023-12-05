import torch


class OpinionWordMiner:
    """
    The class is written for a batch_size 1 scenario
    TODO: Convert to a batch processing scenario 
    """
    def __init__(self, tokenizer, model, gamma = 0.0005, 
                 we_layer_list=[-1], score_by='attention'):
        self.tokenizer = tokenizer
        self.model = model
        self.gamma = gamma
        self.token_level_embeddings = None
        self.score_by = 'attentions' if score_by == 'attention' else 'hidden_states'


    def _get_word_embeddings(self, tokenized_text):
        """
        This function extracts the word/token embeddings from the specified hidden layers.
        """
        tokenized_sentence_output = self.model(**tokenized_text)
        tokenized_sentence_hidden_states = torch.stack(tokenized_sentence_output[self.score_by], dim=0)
        tokenized_sentence_hidden_states = torch.squeeze(tokenized_sentence_hidden_states, dim=1)
        if self.score_by == 'attentions':
            # Extract the lower layer attention heads
            tokenized_sentence_embeddings = tokenized_sentence_hidden_states[2] # (Num Heads, Seq Length, Seq Length)
        else:
            tokenized_sentence_embeddings = tokenized_sentence_hidden_states.permute(1,0,2) # (Num Layers, Seq Length, Emb_Size)


        if self.score_by == 'attentions':
            # Extract specific specific heads for attention based scoring
            self.token_level_embeddings = tokenized_sentence_embeddings.mean(dim=0).detach().numpy()
        else:
            # Extract specific layer outputs for embedding based scoring
            token_vecs_cat = []
            for token in tokenized_sentence_embeddings:
                cat_vec = torch.cat((
                    token[0],
                    token[1],
                    token[2],
                    # token[-1]
                ), dim=0)
                token_vecs_cat.append(cat_vec.detach().numpy())
            self.token_level_embeddings = np.array(token_vecs_cat)
        return self.token_level_embeddings, tokenized_sentence_output
    

    def _get_aligned_subwords_embeddings(self, text):
        """
        Return word embeddings from hidden states when word piece/byte pair tokenizer is used.
        This method aligns the subwords into words by averaging the embeddings of subwords together
        """
        word_embeddings_aligned_list = []
        index_handler_for_cols = []
        tokens = self.tokenizer.tokenize(text)
        new_tokens = word_tokenize(text)

        print("TOKENS ORIGINAL: ", tokens)
        print("TOKENS NEW: ", new_tokens)
        
        for word in new_tokens:
            tokenized_token = self.tokenizer.tokenize(word)
            start_idx = tokens.index(tokenized_token[0])
            end_idx = start_idx + len(tokenized_token)
            word_embeddings = self.token_level_embeddings[start_idx:end_idx]
            # if word == 'car':
            #     print("ll: ", word_embeddings)
            if word_embeddings.shape[0] > 1:
                word_embeddings =np.mean(word_embeddings, axis=0).reshape(1, -1)
                index_handler_for_cols.append([start_idx, end_idx])

            word_embeddings_aligned_list.append(word_embeddings)

        word_embeddings_aligned = np.array(word_embeddings_aligned_list).squeeze(axis=1)

        if self.score_by == 'attentions':
            diff_idx_ = 0
            for start_idx_, end_idx_ in index_handler_for_cols:
                start_idx_-=diff_idx_
                end_idx_-=diff_idx_
                mean_val = np.mean(word_embeddings_aligned[:, start_idx_:end_idx_], axis=1, keepdims=True)
                word_embeddings_aligned[:, [start_idx_]] = mean_val
                word_embeddings_aligned = np.delete(word_embeddings_aligned, np.s_[start_idx_+1:end_idx_], axis=1)
                diff_idx_ = end_idx_ - start_idx_-1

        assert len(new_tokens) == word_embeddings_aligned.shape[0]
        
        if self.score_by == 'attentions':
            assert len(new_tokens) == word_embeddings_aligned.shape[1]

        return torch.tensor(word_embeddings_aligned), new_tokens
    

    def _filter_candidates(self, dataframe, shift_index_filter, pos_filter):
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
        comp_phrase_record = []
        for idx in dataframe[filter_condition].index:
            if (idx + 1) < len(dataframe):
                comp_phrase_record = [
                    ' '.join([dataframe.loc[idx-idx_val]['opinion_word'] for idx_val in shift_index_filter]),
                    dataframe.loc[idx]['attention_score'],
                    '-'.join([dataframe.loc[idx-idx_val]['pos'] for idx_val in shift_index_filter]),
                    dataframe.loc[idx]['dep']
                    ]
                for idx_val in shift_index_filter:
                    compound_phrase_idx.append(idx-idx_val)
                dataframe.loc[len(dataframe)] = comp_phrase_record
        dataframe.drop(index=compound_phrase_idx, inplace=True)
        return dataframe

    
    def mine_opinion_words(self, text, aspect_word, display_df = False):
        # Tokenize the text
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')

        # Query the word embeddings/attention_weights for each token
        word_embeddings, temp_ = self._get_word_embeddings(tokenized_text)

        # Align the word embeddings if the tokenizer splits words into sub words
        word_embeddings, aligned_tokens = self._get_aligned_subwords_embeddings(text)

        # Extract POS tags and Dependency tags
        spacy_tokens_pos_tags = [token.pos_ for token in nlp(text)]
        spacy_tokens_deps = [token.dep_ for token in nlp(text)]

        try:
            # Set default attention score for the aspect word
            aspect_word_score = [0 for i in range(len(aligned_tokens))]

            if self.score_by != 'attentions':
                # Only use RBF kernel similarity if it is embedding based method
                self_attention_matrix = rbf_kernel(word_embeddings, word_embeddings, self.gamma)
            else:
                # RBF Kernel is not required since attention weights already show where the Query is attending to other Key vectors
                self_attention_matrix = word_embeddings

            if ' ' in aspect_word:
                drop_records = []
                aspect_words = aspect_word.split()
                combined_aspect_word_embeddings = []

                for aspect_word in aspect_words:
                    aspect_word_idx = aligned_tokens.index(aspect_word)
                    drop_records.append(aspect_word_idx)
                    combined_aspect_word_embeddings.append(self_attention_matrix[aspect_word_idx])
                aspect_word_score = np.mean(combined_aspect_word_embeddings, axis=0)
            else:
                aspect_word_idx = aligned_tokens.index(aspect_word)
                drop_records = [aspect_word_idx]
                aspect_word_score = self_attention_matrix[:, aspect_word_idx]

            dep_df = pd.DataFrame(aspect_word_score, columns = ['attention_score'], index=aligned_tokens)
            dep_df['pos'] = spacy_tokens_pos_tags
            dep_df['dep'] = spacy_tokens_deps            
            dep_df = dep_df.reset_index().rename(columns = {'index':'opinion_word'})
            dep_df.drop(index=drop_records, inplace=True) # Remove aspect word scores (since it will be highest)

            # Reset index before applying rule based filtering
            dep_df.reset_index(drop=True, inplace=True)
            display(dep_df)

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
        
            # RULE 4: COMPOUND PHRASE EXTRACTION -> Adverbial Phrase (AdP + NOUN)
            dep_df = self._filter_candidates(dataframe=dep_df, 
                                             shift_index_filter=[0, -1], 
                                             pos_filter=['ADP', 'NOUN'])

            if display_df:
                display(dep_df)

            dep_df.reset_index(drop=True, inplace=True)

            # Final filters
            dep_df = dep_df[(dep_df['pos'] == 'ADJ') | \
                            (dep_df['pos'] == 'ADJ-NOUN') |\
                            (dep_df['pos'] == 'ADJ-NOUN-NOUN') |\
                            (dep_df['pos'] == 'ADV-ADJ') |\
                            (dep_df['pos'] == 'ADP-NOUN')
                            ]

            if display_df:
                display(dep_df)
            if dep_df.shape[0] == 0:
                return ''

            # Candidate Reweighting
            opinion_word = dep_df.sort_values(by = 'attention_score', ascending = False).head(1)['opinion_word'].values[0]
            return opinion_word, None
        except:
            return 'NoOptinionTerm', None