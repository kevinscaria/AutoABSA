import pandas as pd
from autoabsa import AutoABSA

# Load data
df = pd.read_json("./Dataset/SemEval14/Train/Laptops_Opinion_Train.json")
df['aspect_term'] = df['aspects'].apply(lambda x: [' '.join(i['term']) for i in x])
df['opinion_word'] = df['opinions'].apply(lambda x: [' '.join(i['term']) for i in x])
df = df.explode(column=['aspect_term', 'opinion_word'])
df = df[['raw_words', 'aspect_term', 'opinion_word']]

# Set Kwargs
kwargs = {
    'gamma': 0.001
}

trainer_kwargs = {
    'output_dir': './outputs/lapt14',
    'overwrite_output_dir': True,
    'num_train_epochs': 8,
    'save_strategy': 'epoch',
    'evaluation_strategy': 'epoch',
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'per_device_train_batch_size': 16,
    'per_device_eval_batch_size': 16,
    'push_to_hub': False,
    'logging_strategy': 'epoch',
    'report_to': 'none',
    'save_total_limit': 1
}

# Instantiate Object
absa = AutoABSA(model_id="bert-base-uncased",
                we_layer_list=[0, 1, 2],
                **kwargs
                )

# Domain Adaptation
absa.fit(docs=df['raw_words'].tolist(),
         **trainer_kwargs
         )
