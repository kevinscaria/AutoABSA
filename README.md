# AutoABSA
Unsupervised Aspect Based Sentiment Analysis

This repository is based on two papers:
1. Aspect Extraction Task - [https://aclanthology.org/2020.acl-main.290/](EmbarrassinglySimpleUnsupervisedAspectExtraction)
2. Opinion Word Extraction and Sentiment POlarity Detection Task - [https://aclanthology.org/2020.acl-main.290/](Embarrassingly Simple Unsupervised Aspect Sentiment Tuple Extraction)

## Usage
The usage is simple

```python
from autoabsa import AutoABSA

input_text = "The chicken was yummy. It was also cheap."
absa = AutoABSA()

aspect_terms_list = absa.get_aspect_term(text=input_text)

for aspect_term in aspect_terms_list:
    opinion_word = absa.get_opinion_word(text=input_text,
                                         aspect="chicken"
                                         )

    sentiment = absa.get_sentiment_polarity(text=input_text,
                                            aspect="chicken"
                                            )

    print(aspect_term, sentiment)
```


