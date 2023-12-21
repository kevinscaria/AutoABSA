import math
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed,
    PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
)
from typing import Union, List
SEED = 42
set_seed(SEED)


class DomainAdaptation:
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 model: PreTrainedModel,
                 text_column: str,
                 chunk_size: int,
                 mlm_proba: float,
                 return_trainer: bool
                 ):
        self.chunk_size = chunk_size
        self.text_column = text_column
        self.model = model
        self.tokenizer = tokenizer
        self.mlm_proba = mlm_proba
        self.return_trainer = return_trainer

    def _tokenize_function(self, examples):
        result = self.tokenizer(examples[self.text_column])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    def _group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.chunk_size) * self.chunk_size
        result = {
            k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def pre_finetune(self,
                     docs: Union[List[str], Dataset, DatasetDict],
                     test_docs: Union[List[str], Dataset],
                     test_size: Union[int, float],
                     **kwargs
                     ):

        if isinstance(docs, list):
            df = pd.DataFrame({self.text_column: docs})
            if test_size is not None:
                train, test = train_test_split(df, random_state=SEED, test_size=test_size)
                dataset = DatasetDict({'train': Dataset.from_pandas(train), 'test': Dataset.from_pandas(test)})
            elif test_size is None and isinstance(test_docs, list):
                test_df = pd.DataFrame({self.text_column: test_docs})
                dataset = DatasetDict({'train': Dataset.from_pandas(df), 'test': Dataset.from_pandas(test_df)})
            else:
                dataset = DatasetDict({'train': Dataset.from_pandas(df), 'test': Dataset.from_pandas(df)})
        elif isinstance(docs, DatasetDict):
            dataset = docs
        else:
            dataset = None

        remove_cols = dataset['train'].column_names
        tokenized_datasets = dataset.map(self._tokenize_function, batched=True, remove_columns=remove_cols)
        lm_datasets = tokenized_datasets.map(self._group_texts, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_proba)

        training_args = TrainingArguments(**kwargs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
            data_collator=data_collator
        )

        eval_results = trainer.evaluate()
        print(f">>> Initial Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        print('Model training started ...')
        trainer.train()
        eval_results = trainer.evaluate()
        print(f">>> Final Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        if self.return_trainer:
            return trainer
