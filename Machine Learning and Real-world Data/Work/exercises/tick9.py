from utils.markov_models import load_bio_data
import os
import random
from tick8 import recall_score, precision_score, f1_score

import numpy as np
import math

from typing import List, Dict, Tuple


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    
    transition_counts = {}
    
    transition_totals = {}
    

    for sequence in hidden_sequences:
        start = sequence[0]
        for i in range(1, len(sequence)):
            current = sequence[i]
            
            if start not in transition_totals:
                transition_totals[start] = 1
            else:
                transition_totals[start] += 1
            
            if (start, current) not in transition_counts:
                transition_counts[(start, current)] = 1
            else:
                transition_counts[(start, current)] += 1
                
            start = current

        if current not in transition_totals:
            transition_totals[current] = 1
        else:
            transition_totals[current] += 1
    
    transition_probs = {}
    
    for key in transition_counts:
        transition_probs[key] = transition_counts[key] / transition_totals[key[0]]
        
    all_keys = list(transition_totals.keys())
    
    for i in range(len(all_keys)):
        for j in range(len(all_keys)):
            if (all_keys[i], all_keys[j]) not in transition_probs:
                transition_probs[(all_keys[i], all_keys[j])] = 0
            
    return transition_probs


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    emission_counts = {}
    
    emission_totals = {}
    
    observed_unique = []
    
    for i in range(len(hidden_sequences)):
        for j in range(len(hidden_sequences[i])):
            hidden = hidden_sequences[i][j]
            observed = observed_sequences[i][j]
            
            if observed not in observed_unique:
                observed_unique.append(observed)
            
            if hidden not in emission_totals:
                emission_totals[hidden] = 1
            else:
                emission_totals[hidden] += 1
                
            if (hidden, observed) not in emission_counts:
                emission_counts[(hidden, observed)] = 1
            else:
                emission_counts[(hidden, observed)] += 1
            
    emission_probs = {}
    
    for key in emission_counts:
        emission_probs[key] = emission_counts[key] / emission_totals[key[0]]
        
    for key in emission_totals:
        for observed in observed_unique:
            if (key, observed) not in emission_probs:
                emission_probs[(key, observed)] = 0
        
    return emission_probs


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

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




def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    
    train = training_data.copy()
    scores = []
    
    for i in range(num_iterations):
        hmm = estimate_hmm_bio(train)
        transition_probs = hmm[0]
        emission_probs = hmm[1]
        
        train = []
        for sequence in unlabeled_data:
            hidden_sequence = viterbi_bio(sequence, transition_probs, emission_probs)
            train.append({'observed': sequence, 'hidden': hidden_sequence})
            
        train.extend(training_data)
        
        #test on dev
    
        predictions = []
        truths = []
        for sample in dev_data:
            prediction = viterbi_bio(sample['observed'], transition_probs, emission_probs)
            
            truth = sample['hidden']
            
            pred_binarized = [1 if x == 'M' else 0 for x in prediction]
            truth_binarized = [1 if x == 'M' else 0 for x in truth]
            
            predictions.append(pred_binarized)
            truths.append(truth_binarized)
        
        recall = recall_score(predictions, truths)
        precision = precision_score(predictions, truths)
        f1 = f1_score(predictions, truths)
        
        scores.append({'recall': recall, 'precision': precision, 'f1': f1})
        
    return scores
            
            
            
            
        
        
        
        
        
        



def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    
    
    recall = [x['recall'] for x in score_list]
    precision = [x['precision'] for x in score_list]
    f1 = [x['f1'] for x in score_list]
    
    clean_plot()
    
    chart_plot([(i, recall[i]) for i in range(len(recall))], 'Recall vs Iteration', 'Iteration', 'Recall')
    chart_plot([(i, precision[i]) for i in range(len(precision))], 'Precision vs Iteration', 'Iteration', 'Precision')
    chart_plot([(i, f1[i]) for i in range(len(f1))], 'F1 vs Iteration', 'Iteration', 'F1')
    


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)



if __name__ == '__main__':
    main()
