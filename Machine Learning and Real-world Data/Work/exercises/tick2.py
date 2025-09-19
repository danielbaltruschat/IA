from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from tick1 import accuracy, predict_sentiment, read_lexicon

import math
import numpy as np


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    
    pos = 0
    neg = 0
    for i in training_data:
        if i['sentiment'] == 1:
            pos += 1
        else:
            neg += 1
    
    return({1: math.log(pos) - math.log(len(training_data)), -1: math.log(neg) - math.log(len(training_data))})


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    
    pos_token_counts = {}
    neg_token_counts = {}
    
    for i in training_data:
        if i['sentiment'] == 1:
            for token in i['text']:
                if token in pos_token_counts:
                    pos_token_counts[token] += 1
                else:
                    pos_token_counts[token] = 1
                    
                if token not in neg_token_counts:
                    neg_token_counts[token] = 0
        else:
            for token in i['text']:
                if token in neg_token_counts:
                    neg_token_counts[token] += 1
                else:
                    neg_token_counts[token] = 1
            
                if token not in pos_token_counts:
                    pos_token_counts[token] = 0
    
    token_ps_pos = {}
    
    log_pos_total = math.log(sum(pos_token_counts.values()))
    log_neg_total = math.log(sum(neg_token_counts.values()))
    
    for token in pos_token_counts:
        if pos_token_counts[token] == 0:
            continue
        else:
            token_ps_pos[token] = math.log(pos_token_counts[token]) -  log_pos_total
    
    token_ps_neg = {}
    for token in neg_token_counts:
        if neg_token_counts[token] == 0:
            continue
        else:
            token_ps_neg[token] = math.log(neg_token_counts[token]) -  log_neg_total
                
    return({1: token_ps_pos, -1: token_ps_neg})

def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    pos_token_counts = {}
    neg_token_counts = {}
    
    for i in training_data:
        if i['sentiment'] == 1:
            for token in i['text']:
                if token in pos_token_counts:
                    pos_token_counts[token] += 1
                else:
                    pos_token_counts[token] = 1
                    
                if token not in neg_token_counts:
                    neg_token_counts[token] = 0
        else:
            for token in i['text']:
                if token in neg_token_counts:
                    neg_token_counts[token] += 1
                else:
                    neg_token_counts[token] = 1
            
                if token not in pos_token_counts:
                    pos_token_counts[token] = 0
    
    token_ps_pos = {}
    
    log_pos_total = math.log(sum(pos_token_counts.values()) + len(pos_token_counts))
    log_neg_total = math.log(sum(neg_token_counts.values()) + len(neg_token_counts))
    
    for token in pos_token_counts:
        token_ps_pos[token] = math.log(pos_token_counts[token] + 1) -  log_pos_total
    
    token_ps_neg = {}
    for token in neg_token_counts:
        token_ps_neg[token] = math.log(neg_token_counts[token] + 1) -  log_neg_total
                
    return({1: token_ps_pos, -1: token_ps_neg})


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    
    p_pos = class_log_probabilities[1]
    p_neg = class_log_probabilities[-1]
        
    for token in review:
        try:
            p_pos += log_probabilities[1][token]
        except:
            pass
        try:
            p_neg += log_probabilities[-1][token]
        except:
            pass
    
    return 1 if p_pos > p_neg else -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")


if __name__ == '__main__':
    main()
