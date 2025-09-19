from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
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
        
    
    

    
    
    

def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
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


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()
