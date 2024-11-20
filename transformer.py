# transformer.py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import optim, sqrt
import matplotlib.pyplot as plt
from typing import List
from utils import *
# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads, causalMask):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3,
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        :param num_heads: number of attention heads for multi-head attention
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.transformer_layers=nn.ModuleList([TransformerLayer(d_model, d_internal, num_heads=num_heads,causalMask=causalMask) for _ in range(num_layers)] )
        self.output_layer = nn.Linear(d_model, num_classes)
    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        x=self.embedding(indices)   # x.shape = (batch_size,seq_len,d_model)
        x=self.positional_encoding(x) # need to verify(same shape as last one)
        #attention_maps are essentially attention_weights
        attention_maps=[]
        for layer in self.transformer_layers:
          x, att_maps = layer(x)
          attention_maps.append(att_maps)
        # Final output layer
        logits = self.output_layer(x)  # Shape: (batch_size, seq_len, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  #log_probs.shape=(batch_size, seq_len, num_classes)  
        return log_probs, attention_maps  


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal, num_heads, causalMask):
        super().__init__()
        self.causalMask=causalMask
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.d_heads = d_internal // num_heads
        self.causalMask = causalMask
        # Define layers
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_model, d_internal)
        self.value = nn.Linear(d_model, d_internal)
        self.output_linear = nn.Linear(d_internal, d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, input_vecs):
        """ Returns the output from the encoder part of the transformer and attention weights """
        # Check the input shape and handle single-batch and multi-batch cases
        if input_vecs.dim() == 2:
            # Case for batch size of 1
            seq_length, d_model = input_vecs.size()
            batch_size = 1
            input_vecs = input_vecs.unsqueeze(0)  # Add a batch dimension
        else:
            # Case for multi-batch inputs
            batch_size, seq_length, d_model = input_vecs.size()

        # Ensure the input dimension matches the model's expected dimension
        assert d_model == self.d_model, "Input dimension must match d_model."

        # Compute queries, keys, and values
        Q = self.query(input_vecs).view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)
        K = self.key(input_vecs).view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)
        V = self.value(input_vecs).view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)

        # Reshape Q, K, and V to (batch_size * num_heads, seq_length, d_heads)
        Q = Q.contiguous().view(batch_size * self.num_heads, seq_length, self.d_heads)
        K = K.contiguous().view(batch_size * self.num_heads, seq_length, self.d_heads)
        V = V.contiguous().view(batch_size * self.num_heads, seq_length, self.d_heads)

        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.d_heads)
        if self.causalMask:
          causal_mask = torch.triu(torch.ones(seq_length,seq_length),diagonal=1).to(scores.device)
          #to(scores.device), Ensures that the mask is created on the same device as scores
          causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))
          scores = scores.view(batch_size, self.num_heads, seq_length, seq_length) + causal_mask
          scores = scores.view(batch_size * self.num_heads, seq_length, seq_length)

        att_maps = torch.softmax(scores, dim=-1)
        # Apply attention weights to values
        attention_output = torch.bmm(att_maps, V)
        # Reshape attention_output back to (batch_size, num_heads, seq_length, d_heads)
        attention_output = attention_output.view(batch_size, self.num_heads, seq_length, self.d_heads)
        attention_output = attention_output.transpose(1, 2)  # (batch_size, seq_length, num_heads, d_heads)
        attention_output = attention_output.contiguous().view(batch_size, seq_length, self.d_internal)
        # Apply the output linear layer and add residual connection
        attention_output = self.output_linear(attention_output)
        out = attention_output + input_vecs  # First residual connection        
        # Feedforward layer with residual connection
        out = self.feedforward(out) + out  # Second residual connection
        # Remove the added batch dimension if batch_size was 1
        if batch_size == 1:
            out = out.squeeze(0)
            att_maps = att_maps.view(self.num_heads, seq_length, seq_length)
        else:
            att_maps = att_maps.view(batch_size, self.num_heads, seq_length, seq_length)
        return out, att_maps
# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

def train_classifier(args, train, dev):
    # Initialize model
    model = Transformer(
        vocab_size=args.vocab_size,
        num_positions=args.num_positions,
        d_model=args.d_model,
        d_internal=args.d_internal,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        causalMask=args.causalMask)
    model.train()    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()    
    num_epochs = 10
    best_dev_accuracy = 0.0  # To keep track of the best validation accuracy
    best_model = None

    # Loop over epochs
    for epoch in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)  # Seed for random shuffling
        ex_idxs = list(range(len(train)))

        # Randomly sample 500 examples for gradient update
        for _ in range(700):  # For each epoch, select 500 random examples
            random.shuffle(ex_idxs)  # Shuffle the examples each time

            ex_idx = ex_idxs[0]  # Pick the first example after shuffle
            example = train[ex_idx]  # Get the example

            inputs = example.input_tensor  # Input tensor (move to GPU if necessary)
            targets = example.output_tensor  # Target tensor (move to GPU if necessary)

            # Forward pass
            log_probs, attention_maps = model(inputs)  # Get log-probabilities and attention maps
            log_probs = log_probs.view(-1, log_probs.size(-1))  # Reshape for bulk loss calculation
            targets = targets.view(-1)  # Flatten targets for compatibility with NLLLoss
            
            # Calculate loss over the selected 500 examples
            loss = loss_fcn(log_probs, targets)

            # Backpropagation and optimization step
            optimizer.zero_grad()  # Reset previous gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights

            # Accumulate loss for this epoch
            loss_this_epoch += loss.item()

        # Print loss for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_this_epoch:.4f}")
        # Evaluate on dev data after each epoch
        model.eval()  # Set model to evaluation mode
        dev_accuracy = evaluate_on_dev(model, dev)
        print(f"Epoch {epoch + 1}/{num_epochs}, Dev Accuracy: {dev_accuracy:.4f}")
        
        # Save the best model based on dev accuracy
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model = model.state_dict()  # Save the model state dictionary

        model.train()  # Switch back to training mode
    # After training, switch to evaluation mode
    # Load the best model before returning
    model.load_state_dict(best_model)
    return model
def evaluate_on_dev(model, dev):
    num_correct = 0
    num_total = 0
    with torch.no_grad():  # Disable gradient calculations
        for example in dev:
            inputs = example.input_tensor
            targets = example.output_tensor
            log_probs, _ = model(inputs)
            predictions = torch.argmax(log_probs, dim=-1)
            num_correct += (predictions == targets).sum().item()
            num_total += targets.size(0)
    accuracy = num_correct / num_total
    return accuracy
"""
def train_classifier(args, train, dev):
    # Initialize model
    model = Transformer(
        vocab_size=args.vocab_size,
        num_positions=args.num_positions,
        d_model=args.d_model,
        d_internal=args.d_internal,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model.train()    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()    
    num_epochs = 3
    best_dev_accuracy = 0.0  # To keep track of the best validation accuracy
    best_model = None
    
    for epoch in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)
        # Shuffle examples
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)        
        for ex_idx in ex_idxs:
            # Extract input and target from LetterCountingExample instance
            example = train[ex_idx]
            inputs = example.input_tensor  # Move inputs to GPU if applicable
            targets = example.output_tensor  # Move targets to GPU if applicable

            # Forward pass
            log_probs, attention_maps = model(inputs)  # Get log-probabilities and attention maps
            log_probs = log_probs.view(-1, log_probs.size(-1))  # Reshape for bulk loss calculation
            targets = targets.view(-1)  # Flatten targets for compatibility with NLLLoss
            
            # Calculate bulk loss over the entire sequence
            loss = loss_fcn(log_probs, targets)
            
            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch
            loss_this_epoch += loss.item()
        
        # Print training loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss_this_epoch:.4f}")
        
        # Evaluate on dev data after each epoch
        model.eval()  # Set model to evaluation mode
        dev_accuracy = evaluate_on_dev(model, dev)
        print(f"Epoch {epoch + 1}/{num_epochs}, Dev Accuracy: {dev_accuracy:.4f}")
        
        # Save the best model based on dev accuracy
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_model = model.state_dict()  # Save the model state dictionary

        model.train()  # Switch back to training mode
    
    # Load the best model before returning
    model.load_state_dict(best_model)
    return model
"""

"""
def train_classifier(args, train, dev):
    # Initialize model
    model = Transformer(
        vocab_size=args.vocab_size,
        num_positions=args.num_positions,
        d_model=args.d_model,
        d_internal=args.d_internal,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model.train()    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fcn = nn.NLLLoss()    
    num_epochs = 10
    for epoch in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)
        # Shuffle examples
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)        
        for ex_idx in ex_idxs:
            # Extract input and target from LetterCountingExample instance
            # from letter_counting.py train=train_bundles
            # train_bundles[ex_idx]=LetterCountingExample(ex_idx, get_letter_count_output(ex_idx,count_only_previous), vocab_index) 
            example = train[ex_idx]
            inputs = example.input_tensor  # Move inputs to GPU if applicable
            targets = example.output_tensor  # Move targets to GPU if applicable
            # inputs.size=(seq_len,), output.size=(seq,)     
            # Forward pass
            log_probs, attention_maps = model(inputs)  # Get log-probabilities and attention maps
            log_probs = log_probs.view(-1, log_probs.size(-1))  # Reshape for bulk loss calculation
            targets = targets.view(-1)  # Flatten targets for compatibility with NLLLoss
            
            # Calculate bulk loss over the entire sequence
            loss = loss_fcn(log_probs, targets)
            
            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for the epoch
            loss_this_epoch += loss.item()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_this_epoch:.4f}")
    
    # Switch to evaluation mode
    model.eval()
    return model

####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100: 
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
"""