from utils.sentiment_detection import clean_plot, read_tokens, chart_plot, best_fit
from typing import List, Tuple, Callable
import os
import math


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    
    temp = best_fit(token_frequencies_log, token_frequencies)
    m = temp[0]
    c = temp[1]
    
    def f(rank):
        log_freq = m * math.log(rank) + c
        return math.exp(log_freq)
    
    return f
    
    


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    
    token_freqs = {}
    
    for file in os.listdir(dataset_path):
        for token in read_tokens(os.path.join(dataset_path, file)):
            try:
                token_freqs[token] += 1
            except KeyError:
                token_freqs[token] = 1
    
    return sorted([(k, v) for k, v in token_freqs.items()], key=lambda x: x[1], reverse=True)
            


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    
    ranks = range(1, 10001)
    freqs = [f[1] for f in frequencies[:10000]]
    
    chart_plot(list(zip(ranks, freqs)), "Word Frequencies", "Rank", "Frequency")
    
    


def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    
    with open("exercises/words.txt", "r") as f:
        words = [line.strip() for line in f]
        
    ranks = []
    freqs = []
    for word in words:
        for i in range(len(frequencies)):
            if frequencies[i][0] == word:
                ranks.append(i)
                freqs.append(frequencies[i][1])
                break
            
    chart_plot(list(zip(ranks, freqs)), "Word Frequencies", "Rank", "Frequency")
    
    
        


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    
    ranks = range(1, 10001)
    freqs = [f[1] for f in frequencies[:10000]]
    
    zipped = list(zip(ranks, freqs))
    log_zipped = list(map(lambda x: (math.log(x[0]), math.log(x[1])), zipped))
    
    f = estimate_zipf(log_zipped, zipped)
    
    lobf = [(x, f(x)) for x in range(1, 10001)]
    log_lobf = list(map(lambda x: (math.log(x[0]), math.log(x[1])), lobf))
    
    chart_plot(log_zipped, "Zipf's Law Logs", "log(rank)", "log(frequency)")
    chart_plot(log_lobf, "Zipf's Law Logs", "log(rank)", "log(frequency)")
    
    
    


def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    word_freqs = {}
    word_count = 0
    two_pow = 1
    
    #list of tuples
    datapoints = []
    
    for file in os.listdir(dataset_path):
        for token in read_tokens(os.path.join(dataset_path, file)):
            word_count += 1
            try:
                word_freqs[token] += 1
            except:
                word_freqs[token] = 1
            
            if word_count == two_pow:
                datapoints.append((word_count, len(word_freqs)))
                two_pow *= 2
                
    datapoints.append((word_count, len(word_freqs)))
    
    return datapoints


def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    
    log_data = list(map(lambda x: (math.log(x[0]), math.log(x[1])), type_counts))
    chart_plot(log_data, "Heap's Law", "log(Tokens)", "log(Types)")


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
