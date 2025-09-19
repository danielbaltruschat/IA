import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from tick5 import generate_random_cross_folds, cross_validation_accuracy

import math

def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    
    pos = 0
    neg = 0
    neutral = 0
    
    for i in training_data:
        if i['sentiment'] == 1:
            pos += 1
        elif i['sentiment'] == -1:
            neg += 1
        elif i['sentiment'] == 0:
            neutral += 1
    
    return({1: math.log(pos) - math.log(len(training_data)), -1: math.log(neg) - math.log(len(training_data)), 0: math.log(neutral) - math.log(len(training_data))})


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    
    pos_token_counts = {}
    neg_token_counts = {}
    neutral_token_counts = {}
    
    for i in training_data:
        if i['sentiment'] == 1:
            for token in i['text']:
                if token in pos_token_counts:
                    pos_token_counts[token] += 1
                else:
                    pos_token_counts[token] = 1
                    
                if token not in neg_token_counts:
                    neg_token_counts[token] = 0
                    
                if token not in neutral_token_counts:
                    neutral_token_counts[token] = 0
                    
        elif i['sentiment'] == -1:
            for token in i['text']:
                if token in neg_token_counts:
                    neg_token_counts[token] += 1
                else:
                    neg_token_counts[token] = 1
            
                if token not in pos_token_counts:
                    pos_token_counts[token] = 0
                    
                if token not in neutral_token_counts:
                    neutral_token_counts[token] = 0
        else:
            for token in i['text']:
                if token in neutral_token_counts:
                    neutral_token_counts[token] += 1
                else:
                    neutral_token_counts[token] = 1
            
                if token not in pos_token_counts:
                    pos_token_counts[token] = 0
                if token not in neg_token_counts:
                    neg_token_counts[token] = 0
    
    token_ps_pos = {}
    
    log_pos_total = math.log(sum(pos_token_counts.values()) + len(pos_token_counts))
    log_neg_total = math.log(sum(neg_token_counts.values()) + len(neg_token_counts))
    log_neutral_total = math.log(sum(neutral_token_counts.values()) + len(neutral_token_counts)) 
    
    
    for token in pos_token_counts:
        token_ps_pos[token] = math.log(pos_token_counts[token] + 1) -  log_pos_total
    
    token_ps_neg = {}
    for token in neg_token_counts:
        token_ps_neg[token] = math.log(neg_token_counts[token] + 1) -  log_neg_total
       
    token_ps_neutral = {}
    for token in neutral_token_counts:
        token_ps_neutral[token] = math.log(neutral_token_counts[token] + 1) -  log_neutral_total
             
    return({1: token_ps_pos, -1: token_ps_neg, 0: token_ps_neutral})
    

def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    
    correct = 0
    total = 0
    
    for i in range(len(true)):
        if true[i] == pred[i]:
            correct += 1
        total += 1
        
    return correct / total


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    
    p_pos = class_log_probabilities[1]
    p_neg = class_log_probabilities[-1]
    p_neutral = class_log_probabilities[0]
    
    for token in review:
        try:
            p_pos += log_probabilities[1][token]
        except:
            pass
        try:
            p_neg += log_probabilities[-1][token]
        except:
            pass
        try:
            p_neutral += log_probabilities[0][token]
        except:
            pass
        
    if p_pos > p_neg:
        if p_pos > p_neutral:
            return 1
        else:
            return 0
    else:
        if p_neg > p_neutral:
            return -1
        else:
            return 0
        


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    
    key1 = list(agreement_table.keys())[0]
    k = agreement_table[key1][1] + agreement_table[key1][-1] + agreement_table[key1][0]
    
    N = len(agreement_table)
    
    p_e = 0
    
    for j in range(-1, 2):
        total = 0
        for key in agreement_table:
            val = agreement_table[key][j]
            total += val
        p_e += (val / (N*k)) ** 2
    
    p_a = 0
    
    for key in agreement_table:
        total = 0
        for j in range(-1, 2):
            val = agreement_table[key][j]
            total += val * (val - 1)
        p_a += total / (k * (k - 1))
    
    p_a /= N
            
    return (p_a - p_e) / (1 - p_e)
    


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    agreement_table = {}
    
    reviews = list(review_predictions[0].keys())
    for review in reviews:
        agreement_table[review] = {1: 0, -1: 0, 0: 0}
    
    for i in range(len(review_predictions)):
        for review in reviews:
            sentiment = review_predictions[i][review]
            agreement_table[review][sentiment] += 1
    
    return agreement_table


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2022.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
