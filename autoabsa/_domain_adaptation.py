from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, 
    Trainer, TrainingArguments
)

class DomainAdaptation:
    def __init__(self, text_input, 
                 text_column = 'text', 
                 seed = 42, 
                 test_size = None, 
                 chunk_size=128, 
                 model_ckpt='bert-large-uncased'):
        self.seed = 42
        self.test_size = test_size
        self.chunk_size = chunk_size
        self.text_column = text_column

        # Load the BERT-large tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForMaskedLM.from_pretrained(model_ckpt)

        if isinstance(text_input, list):
            df = pd.DataFrame({text_column:text_input})
            if test_size is not None:
                train, test = train_test_split(df, random_state = self.seed, test_size = self.test_size)
                self.dataset = DatasetDict({'train':Dataset.from_pandas(train), 'test':Dataset.from_pandas(test)})
            else:
                self.dataset = DatasetDict({'train':Dataset.from_pandas(df), 'test':Dataset.from_pandas(df)})
        else:
            self.dataset = text_input


    def tokenize_function(self, examples):
        result = self.tokenizer(examples[self.text_column])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result


    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.chunk_size) * self.chunk_size
        result = {
            k: [t[i : i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    def pre_finetune(self, root_path, mlm_proba = 0.15, batch_size=16, epochs = 8, return_trainer = False):
        remove_cols = self.dataset['train'].column_names
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True, remove_columns=remove_cols)
        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_proba)

        training_args = TrainingArguments(
            output_dir=root_path,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=False,
            fp16=True,
            logging_strategy='epoch',
            save_total_limit = 2,
            load_best_model_at_end=True 
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=lm_datasets["train"],
            eval_dataset=lm_datasets["test"],
            data_collator=data_collator
        )

        eval_results = self.trainer.evaluate()
        print(f">>> Initial Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
        print('Model training started ...')
        self.trainer.train()
        eval_results = self.trainer.evaluate()
        print(f">>> Final Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

        if return_trainer:
            return self.trainer