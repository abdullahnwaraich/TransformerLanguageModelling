# lm.py
import sys
import argparse
import json
import time
from transformer_lm import *
from utils import * 
import numpy as np
####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################
# to make it pvt do _parse_args()
def parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    # Strip out the extra arguments passed by Jupyter/Colab
    sys.argv = sys.argv[:1]  # Keep only the first argument, i.e., the script name


    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--model', type=str, default='NEURAL', help='model to run (UNIFORM or NEURAL)')
    parser.add_argument('--train_path', type=str, default='data/text8-100k.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/text8-dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_path', type=str, default='data/text8-test.txt', help='path to test set')
    parser.add_argument('--output_bundle_path', type=str, default='models/output.json', help='path to write the results json to (you should not need to modify)')
    parser.add_argument('--vocab_size', type=int, default=27, help="Vocabulary size")  
    parser.add_argument('--d_model', type=int, default=120 ,help="embedding dimensions")  
    parser.add_argument('--d_internal', type=int, default=90 ,help="query/key dimensions")  
    parser.add_argument('--num_classes', type=int, default=27, help="number of output classes")
    parser.add_argument('--num_positions', type=int, default=20, help="seq_len")
    parser.add_argument('--num_heads', type=int, default=2, help="Number of attention heads")      
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the Transformer")  
    parser.add_argument('--batch_size', type=int, default=12, help="Size of batches")  
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")  
    parser.add_argument('--lr', type=int, default=1e-4, help="Learning rate")  
    parser.add_argument('--causalMask', type=bool, default=True, help="causla mask for attention")  
    args = parser.parse_args()
    return args
def read_text(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_text = ""
    for line in open(file):
        all_text += line
    print("%i chars read in" % len(all_text))
    return all_text
     
def run_sanity_check(lm, vocab_index):
    """
    Runs two sanity checks: (1) The language model must return valid probabilities for a few contexts. This checks that
    your model can take sequences of different lengths and contexts of different lengths without crashing.
    (2) Your reported next character distribution must agree with get_log_prob_sequence
    :param lm: the trained LM
    :return: True if the output is sane, false otherwise
    """
    contexts = [" ", " a person ", " some person "]
    next_seqs = ["s", "sits", "stands"]
    sane = True
    for context in contexts:
        for next_seq in next_seqs:
            log_prob = lm.get_log_prob_sequence(next_seq, context)
            if log_prob > 0.0:
                sane = False
                print("ERROR: sanity checks failed, LM log probability %f is invalid" % (log_prob))
            log_prob_from_single_probs = 0.0
            for i in range(0, len(next_seq)):
                # print(repr(context + next_seq[0:i]))
                # print(repr(next_seq[i]))
                next_char_log_probs = lm.get_next_char_log_probs(context + next_seq[0:i])
                # print(repr(next_char_log_probs))
                log_prob_from_single_probs += next_char_log_probs[vocab_index.index_of(next_seq[i])]
            if abs(log_prob_from_single_probs - log_prob) > 1e-3:
                sane = False
                print("ERROR: sanity checks failed, LM prob from sequence and single characters disagree: %f %f" % (log_prob, log_prob_from_single_probs))
    return sane



def normalization_test(lm, vocab_index):
    """
    Tests that LM normalizes, checks multiple contexts and sums over everything in the vocabulary to make sure it
    sums to one
    :param lm:
    :param voc:
    :return:
    """
    contexts = [" ", " a person "]
    normalizes = True
    for context in contexts:
        total_prob_single = np.sum(np.exp(lm.get_next_char_log_probs(context)))
        if total_prob_single < 0.99 or total_prob_single > 1.01:
            normalizes = False
            print("Probabilities sum to %f not 1.0 for context %s" % (total_prob_single, context))
        total_prob_seq = 0.0
        for char_idx in range(0, len(vocab_index)):
            total_prob_seq += np.exp(lm.get_log_prob_sequence(vocab_index.get_object(char_idx), context))
        if total_prob_seq < 0.99 or total_prob_seq > 1.01:
            normalizes = False
            print("Probabilities sum to %f not 1.0 for context %s" % (total_prob_seq, context))
    return normalizes

def perplexity_range_check(perplexity):
    if perplexity < 3.5:
        print("ERROR: checks failed, the perplexity is too low. Please make sure you are using causal mask and make sure you are scoring the entire next_chars (instead of a single chunk) in get_log_prob_sequence")
        return False
    return True

def LogProbTextForTransformerLM(text,lm, vocab_index):
    if len(text)<=9:
      return float(lm.get_log_prob_sequence(text, ""))

    num_even_sized_chunks = len(text) // 10
    len_last_chunk = len(text)%10
    text_chunks=[text[10*j:10*(j+1)] for j in range(0,num_even_sized_chunks)]
    log_prob = float(lm.get_log_prob_sequence(text[:10], ""))
    for k in range(1,len(text_chunks)):
      log_prob += float(lm.get_log_prob_sequence(text_chunks[k],text_chunks[k-1]))
    if len_last_chunk >=1:
      log_prob += float(lm.get_log_prob_sequence(text[-len_last_chunk:],text_chunks[-1]))
    return log_prob

"""
def beam_search_generate_sequence(lm,start_text,vocab_index,gen_length=40,num_pos=20, beam_size=3):
      
    if len(start_text) >= gen_length:
      raise ValueError("Start text length must be less than gen_length")      
    generated_texts = [start_text for _ in range(beam_size)]
    log_pbs_bm_seqs=[0.0 for _ in range(beam_size)]
    while len(generated_texts[0]) <= num_pos - 1:
      current_contexts = [generated_texts[m] for m in range(0,beam_size)]
    while len(generated_texts[0]) <= gen_length -1:
      current_contexts = [ generated_texts[-num_pos+1:] for _ in range(beam_size)] #paths starting with highest to lowest probability
      log_pbs = lm.get_next_char_log_probs(current_contexts)
      topk_indices = np.argsort(log_pbs)[-beam_size:][::-1] 
      topk_values = log_pbs[topk_indices]
      for j in range(beam_size):
        current_contexts[j] += vocab_index.get_object(topk_indices[j])
        log_pbs_bm_seqs[j] += topk_values[j]  
"""

def nucleus_sampling_generate_sequence(lm, start_text, vocab_index, gen_length=40, num_pos=20, p=0.95):
    """
    Generates a sequence using nucleus sampling (top-p sampling).
    
    Args:
        lm: Language model instance.
        start_text (str): Initial text to start the sequence.
        vocab_index: Object to map between indices and tokens.
        gen_length (int): Total length of the sequence to generate.
        num_pos (int): Context length for the model.
        p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        str: The generated sequence.
    """
    if len(start_text) >= gen_length:
        raise ValueError("Start text length must be less than gen_length")

    generated_text = start_text

    # Generate until the desired length is reached
    for _ in range(gen_length - len(start_text)):
        # Ensure the context does not exceed num_pos
        current_context = generated_text[-num_pos:]

        # Get the log probabilities for the next character
        log_probs = lm.get_next_char_log_probs(current_context)

        # Convert log probabilities to probabilities
        probs = np.exp(log_probs - np.max(log_probs))  # Subtract max log_prob for numerical stability
        probs /= np.sum(probs)  # Normalize to get a valid probability distribution

        # Sort probabilities and indices in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Compute the cumulative sum of probabilities
        cumulative_probs = np.cumsum(sorted_probs)

        # Find the smallest set of tokens whose cumulative probability exceeds p
        cutoff_index = np.searchsorted(cumulative_probs, p)
        nucleus_indices = sorted_indices[:cutoff_index + 1]  # Indices of the nucleus

        # Normalize probabilities within the nucleus
        nucleus_probs = sorted_probs[:cutoff_index + 1]
        nucleus_probs /= np.sum(nucleus_probs)

        # Sample the next character from the nucleus
        next_index = np.random.choice(nucleus_indices, p=nucleus_probs)
        next_char = vocab_index.get_object(next_index)

        # Append the selected character to the generated text
        generated_text += next_char

    return generated_text




def beam_search_generate_sequence(lm, start_text, vocab_index, gen_length=40, num_pos=20, beam_size=10):

    if len(start_text) >= gen_length:
        raise ValueError("Start text length must be less than gen_length")

    # Initialize beams: start with the initial text and log probability of 0.0
    beams = [(start_text, 0.0)]  # (generated_text, cumulative_log_prob)

    # Generate until the desired length is reached
    for _ in range(gen_length - len(start_text)):
        new_beams = []

        for text, log_prob_sum in beams:
            # Ensure the context does not exceed num_pos
            current_context = text[-num_pos:]

            # Get the log probabilities for the next character
            log_probs = lm.get_next_char_log_probs(current_context)

            # Select the top-k most probable next characters
            topk_indices = np.argsort(log_probs)[-beam_size:][::-1]  # Indices of top-k
            topk_values = log_probs[topk_indices]  # Log-probabilities of top-k

            # Update beams with the new sequences and their cumulative log probabilities
            for idx, log_prob in zip(topk_indices, topk_values):
                char = vocab_index.get_object(idx)
                new_text = text + char
                new_log_prob_sum = log_prob_sum + log_prob
                new_beams.append((new_text, new_log_prob_sum))

        # Sort the new beams by their cumulative log probabilities and keep the top-k
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    # Return the sequence with the highest log probability
    return beams[0][0]

def print_evaluation(text, lm, vocab_index, output_bundle_path):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    # since length of test_text is 500 and trained model can only take max len input seq of 20
    # had to condition entire text into length 10 chunks text=(w_1,w_2,...,w_{T}) where each len(w_i)=10
    # P(text)=P(w_{T}|w_{T-1})P(w_{T-1}|w_{T-2})...P(w_{2}|w_{1})  
    sane = run_sanity_check(lm, vocab_index)
    text_chunks=[text[10*j:10*(j+1)] for j in range(0,50)]
    log_prob = float(lm.get_log_prob_sequence(text[:10], ""))
    for k in range(1,len(text_chunks)):
      log_prob += float(lm.get_log_prob_sequence(text_chunks[k],text_chunks[k-1]))
#    for j in range(1,25):
#      log_prob+= float(lm.get_log_prob_sequence(text[20*j:20*(j+1)],text[20*(j-1):20*j]))    
    avg_log_prob = log_prob/len(text)
    perplexity = np.exp(-log_prob / len(text))
    normalizes = normalization_test(lm, vocab_index)
    range_check = perplexity_range_check(perplexity)
    data = {'sane': sane, 'normalizes': normalizes, 'range': range_check, 'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    with open(output_bundle_path, 'w') as outfile:
        json.dump(data, outfile)
    return data


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    save_model_path='/content/drive/MyDrive/a3-distrib/models/transformer_lm_ep_ten.pth'
    train_text = read_text(args.train_path)
    dev_text = read_text(args.dev_path)
    # Vocabs is lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    print("First 100 characters of train:")
    print(train_text[0:100])
    # Train our model
    if args.model == "NEURAL":
        print('implementing neural language model')
        model = train_lm(args, train_text, dev_text, vocab_index,save_model_path)
    elif args.model == "UNIFORM":
        model = UniformLanguageModel(len(vocab))
    else:
        raise Exception("Pass in either UNIFORM or NEURAL to run the appropriate system")

    print_evaluation(dev_text, model, vocab_index, args.output_bundle_path)
