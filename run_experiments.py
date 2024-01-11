import os
import shutil
import json
from autoabsa import AutoABSA
from autoabsa.utils import get_df

model_ckpts = [
    "bert-base-uncased",
    "bert-large-uncased",

    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",

    "facebook/muppet-roberta-base",
    "facebook/muppet-roberta-large",
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",

    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",

    "SpanBERT/spanbert-base-cased",
    "SpanBERT/spanbert-large-cased",

    "microsoft/deberta-v3-xsmall",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge"
]

sample_fracs = [0.05, 0.1, 0.25, 0.5, 1.0]
data_path = f"./Dataset/sample_type/"
dump_path = './outputs/'

save_train_output = False

trainer_kwargs = {
    'overwrite_output_dir': True,
    'num_train_epochs': 5,
    'save_strategy': 'epoch',
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'evaluation_strategy': 'epoch',
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'push_to_hub': False,
    'logging_strategy': 'epoch',
    'report_to': 'none'
}

for model_ckpt in model_ckpts[:]:
    model_name = model_ckpt.split("/")[-1]

    for train_file, test_file in zip(os.listdir(data_path.replace("sample_type", "Test")),
                                     os.listdir(data_path.replace("sample_type", "Test"))):

        train_data_path = os.path.join(data_path.replace("sample_type", "Test"), train_file)
        test_data_path = os.path.join(data_path.replace("sample_type", "Test"), test_file)

        # Load test data
        test_df = get_df(data_path=test_data_path)

        for sample_frac in sample_fracs[:]:

            # Load train data
            train_df = get_df(data_path=train_data_path, sample_size=sample_frac)

            # Create experiment id
            exp_id = f"{model_name}-frac_{sample_frac}-{os.path.splitext(os.path.basename(train_data_path))[0]}"
            exp_id = exp_id.replace("_Test", "")
            print("Experiment ID: ", exp_id)

            status_file_path = os.path.join(dump_path, 'status.json')
            if os.path.exists(status_file_path):
                with open(status_file_path, 'r') as f:
                    status = json.load(f)
                    if status.get(exp_id) is None:
                        status[exp_id] = False

                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2)
            else:
                status = {exp_id: False}
                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2)

            if status.get(exp_id) is None or not status[exp_id]:

                # Set Kwargs
                kwargs = {'gamma': 0.001}
                exp_dump_path = os.path.join(dump_path, exp_id)
                trainer_kwargs['output_dir'] = exp_dump_path

                # Instantiate Object
                absa = AutoABSA(
                    tokenizer_name=model_ckpt,
                    model_id=model_ckpt,
                    we_layer_list=[1, 2],
                    **kwargs
                )

                # Fit - Get Sentiment tuples
                trainer = absa.fit(docs=train_df['raw_words'].tolist(),
                                   return_trainer=True,
                                   **trainer_kwargs
                                   )

                # Transform - Train Set
                if save_train_output:
                    start_idx = 0
                    end_idx = -1
                    res_tr_df = absa.transform(docs=train_df['raw_words'].tolist()[start_idx:end_idx],
                                               extract_aspects=False,
                                               aspect_term_list=train_df['aspect_term'].tolist()[start_idx:end_idx],
                                               extract_opinion_words=True,
                                               fitted_model=trainer.model,
                                               debug=False
                                               )

                    res_tr_df['opinion_word_gt'] = train_df['opinion_word'].tolist()[start_idx:end_idx]
                    res_tr_df['polarity_gt'] = train_df['polarity'].tolist()[start_idx:end_idx]
                    res_tr_df.to_csv(os.path.join(exp_dump_path, 'train_results.csv'))

                # Transform - Test Set
                start_idx = 0
                end_idx = -1
                res_te_df = absa.transform(docs=test_df['raw_words'].tolist()[start_idx:end_idx],
                                           extract_aspects=False,
                                           aspect_term_list=test_df['aspect_term'].tolist()[start_idx:end_idx],
                                           extract_opinion_words=True,
                                           fitted_model=trainer.model,
                                           debug=False
                                           )

                res_te_df['opinion_word_gt'] = test_df.iloc[start_idx:end_idx]['opinion_word'].values
                res_te_df['polarity_gt'] = test_df.iloc[start_idx:end_idx]['polarity'].values
                res_te_df.to_csv(os.path.join(exp_dump_path, 'test_results.csv'))

                for path in os.listdir(exp_dump_path):
                    if not path.endswith(".csv"):
                        shutil.rmtree(os.path.join(exp_dump_path, path), ignore_errors=True)

                # Update Status of experiment
                with open(status_file_path, 'r') as f:
                    status = json.load(f)
                    status[exp_id] = True

                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2)

                print(" ======================== ******* ======================== ")

            else:
                print('Experiment already completed')
                print(" ======================== ******* ======================== ")
