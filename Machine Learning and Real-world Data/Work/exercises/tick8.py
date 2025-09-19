from utils.markov_models import load_dice_data
import os
from tick7 import estimate_hmm
import random

from typing import List, Dict, Tuple

import numpy as np
import math


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    def log(x):
        if x == 0:
            return -math.inf
        else:
            return math.log(x)
    
    states = []
    for key in transition_probs:
        if key[0] not in states:
            states.append(key[0])
    
    trellis = np.zeros((len(observed_sequence), len(states)))
    memo_states = np.zeros((len(observed_sequence) - 1, len(states)), dtype=np.int8)
    
    for i in range(len(observed_sequence)):
        if i == 0:
            for j in range(len(states)):
                p = emission_probs[(states[j], observed_sequence[0])]
                
                trellis[0, j] = log(p)
        
        else:
            for j in range(len(states)):
                max_prob = -math.inf
                max_state = None
                for k in range(len(states)):
                    prob = trellis[i - 1, k] + log(transition_probs[(states[k], states[j])]) + log(emission_probs[(states[j], observed_sequence[i])])
                    if prob >= max_prob:
                        max_prob = prob
                        max_state = k
                trellis[i, j] = max_prob
                memo_states[i - 1, j] = max_state
                
    max_prob = -math.inf
    max_state = None
    for i in range(len(states)):
        prob = trellis[len(observed_sequence) - 1, i]
        if prob >= max_prob:
            max_prob = prob
            max_state = i
    
    hidden_sequence = [states[max_state]]
    current = max_state
    for i in range(len(observed_sequence)-1, 0, -1):
        hidden_sequence.insert(0, states[memo_states[i - 1, current]])
        current = memo_states[i - 1, current]
        
    return hidden_sequence
                
    
    
        


def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    
    true_positives = 0
    total_positives = 0
    
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == 1:
                total_positives += 1
                if pred[i][j] == true[i][j]:
                    true_positives += 1
    
    return true_positives / total_positives


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    
    true_positives = 0
    total_positives = 0
    
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if true[i][j] == 1:
                total_positives += 1
                if pred[i][j] == true[i][j]:
                    true_positives += 1
    
    return true_positives / total_positives


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    
    precision = precision_score(pred, true)
    recall = recall_score(pred, true)
    
    return 2 * (precision * recall) / (precision + recall)


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    
    n = 10
    
    indexes = np.random.permutation(len(data))
    folds = np.array_split(indexes, n)
    for i in range(n):
        folds[i] = [data[j] for j in folds[i]]
    
    
    scores = {'precision': 0, 'recall': 0, 'f1': 0}
    for i in range(n):
        test = folds[i]
        train = []
        for j in range(n):
            if j != i:
                train.extend(folds[j])
                
        hmm = estimate_hmm(train)
        transition_probs = hmm[0]
        emission_probs = hmm[1]
        
        preds = []
        truths = []
        for item in test:
            prediction = viterbi(item['observed'], transition_probs, emission_probs)

            truth = item['hidden']
            
            prediction = [1 if x == 'W' else 0 for x in prediction]
            truth = [1 if x == 'W' else 0 for x in truth]
            
            preds.append(prediction)
            truths.append(truth)
        
        scores['precision'] += precision_score(preds, truths)
        scores['recall'] += recall_score(preds, truths)
        scores['f1'] += f1_score(preds, truths)
        
    scores['precision'] /= n
    scores['recall'] /= n
    scores['f1'] /= n
    
    return scores
        
        
    
    
        
            
            
        
        
        


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    viterbi(['4', '4', '3', '5', '6', '3', '3', '1', '1', '1', '1', '1', '1', '4', '1', '2', '2', '1', '1', '5', '2', '1', '2', '1', '1', '5', '1', '1', '1', '5', '1', '1', '6', '6', '1', '1', '3', '4', '2', '6', '2', '3', '1', '1', '2', '2', '1', '3', '4', '1', '1', '2', '2', '6', '1', '3', '1', '3', '1', '1', '1', '1', '5', '2', '2', '1', '2', '3', '2', '4', '6', '4', '2', '4', '6', '2', '3', '6', '4', '5', '1', '6', '1', '1', '1', '4', '1', '1', '1', '1', '3', '3', '6', '1', '5', '4', '5', '1', '1', '3', '5', '3', '4', '2', '1', '1', '2', '4', '3', '5', '1', '3', '5', '1', '5', '6', '1', '1', '5', '3', '4', '4', '1', '1', '1', '1', '1', '1', '1', '5', '6', '2', '1', '2', '6', '1', '2', '1', '1', '1', '1', '1', '1', '1', '2', '3', '5', '4', '1', '4', '1', '6', '5', '6', '4', '4', '5', '4', '5', '5', '6', '1', '5', '1', '2', '5', '5', '6', '1', '6', '6', '2', '4', '4', '2', '3', '6', '5', '4', '6', '5', '6', '1', '5', '6', '1', '3', '3', '4', '1', '2', '3', '2', '4', '3', '3', '4', '6', '2', '4', '4', '6', '2', '2', '2', '2', '1', '1', '1', '1', '4', '1', '3', '6', '5', '2', '3', '3', '1', '2', '1', '1', '1', '2', '1', '3', '6', '2', '5', '1', '4', '4', '4', '6', '3', '4', '4', '4', '1', '6', '6', '3', '2', '3', '4', '5', '4', '6', '5', '2', '1', '4', '5', '2', '2', '1', '2', '5', '2', '2', '3', '3', '2', '3', '1', '2', '3', '2', '2', '1', '1', '4', '1', '4', '6', '1', '1', '5', '2', '1', '2', '1', '1', '1', '1', '2', '2', '1', '1', '2', '1', '6', '1', '1', '1', '1', '2', '1', '1', '1', '1', '1', '5', '1', '1', '1', '2', '1', '2', '2', '1', '1', '1', '5', '4', '3', '5', '4', '2', '5', '3', '4', '5', '3', '1', '2', '1', '4', '2', '4', '6', '3', '5', '5', '5', '1', '4', '5', '1', '2', '2', '1', '2', '6', '6', '4', '6', '3', '5', '4', '6', '4', '2', '6', '2', '6', '5', '6', '1', '1', '6', '3', '6', '1', '4', '5', '5', '6', '3', '4', '4', '6', '6', '6', '5', '2', '2', '1', '1', '1', '2', '1', '6', '2', '1', '1', '1', '6', '1', '6', '1', '1', '3', '1', '1', '1', '1', '1', '5', '2', '1', '2', '3', '2', '2', '1', '2', '5', '1', '2', '1', '1', '1', '2', '1', '2', '1', '6', '2', '6', '1', '2', '4', '3', '3', '6', '2', '2', '4', '5', '4', '1', '6', '2', '1', '2', '3', '3', '2', '1', '3', '3', '5', '1', '2', '3', '3', '6', '6', '4', '1', '3', '6', '1', '5', '6', '2', '3', '2', '1', '5', '5', '2', '1', '1', '5', '1', '1', '1', '1', '2', '2', '2', '1', '2', '4', '1', '1', '1', '6', '2', '1', '3', '1', '1', '2', '2', '2', '1', '1', '6', '1', '1', '2', '2', '1', '2', '2', '1', '1', '1', '6', '2', '2', '1', '1', '4', '1', '1', '1', '2', '1', '1', '4', '6', '2', '5', '3', '6', '1', '1', '6', '6', '4', '6', '6', '1', '5', '1', '2', '2', '2', '1', '1', '1', '1', '2', '1', '5', '1', '1', '1', '1', '2', '6', '1', '1', '4', '1', '5', '3', '5', '1', '2', '2', '5', '6', '1', '3', '1', '1', '2', '1', '5', '1', '5', '4', '1', '4', '3', '1', '1', '5', '2', '2', '1', '3', '1', '1', '1', '2', '3', '1', '1', '2', '3', '1', '5', '2', '5', '5', '4', '4', '3', '1', '4', '5', '1', '6', '1', '6', '4', '6', '2', '6', '6', '5', '2', '4', '1', '1', '4', '1', '2', '1', '5', '1', '5', '1', '1', '2', '1', '1', '1', '1', '2', '4', '6', '1', '1', '6', '2', '4', '3', '1', '1', '5', '2', '1', '4', '1', '1', '1', '1', '5', '2', '4', '1', '2', '6', '1', '5', '5', '2', '1', '2', '1', '1', '2', '2', '6', '1', '1', '1', '5', '1', '1', '1', '2', '6', '6', '6', '1', '1', '6', '3', '3', '4', '3', '4', '4', '6', '2', '2', '1', '2', '5', '2', '1', '1', '1', '1', '1', '3', '6', '6', '6', '2', '1', '4', '1', '1', '1', '1', '1', '2', '2', '2', '1', '6', '2', '1', '3', '1', '6', '2', '2', '3', '1', '1', '4', '4', '2', '6', '3', '2', '3', '3', '5', '3', '3', '3', '5', '5', '6', '1', '4', '3', '2', '5', '5', '2', '4', '5', '2', '4', '4', '5', '1', '1', '1', '6', '3', '1', '3', '4', '1', '1', '1', '1', '6', '6', '1', '4', '3', '4', '1', '1', '3', '2', '2', '1', '1', '3', '4', '5', '4', '4', '2', '6', '5', '3', '5', '3', '6'], transition_probs, emission_probs)
    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
