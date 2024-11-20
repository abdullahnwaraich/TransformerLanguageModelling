# models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from transformer import Transformer, TransformerLayer, PositionalEncoding
class TransformerLM(Transformer):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads, causalMask,vocab_index):
      super().__init__(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, num_heads, causalMask)
      self.vocab_index=vocab_index
      
    def get_next_char_log_probs(self, context) -> np.ndarray:
#      if context[0] != " ":
#        context = " " + context 

    # Set the model to evaluation mode
      self.eval()
    
    # Initialize the context indices list
      context_indices = []
    
    # Convert each character in the context to an index
      for char in context:
        idx = self.vocab_index.index_of(char)  # Corrected from context to char
        if idx == -1:  # If the character is not in vocabulary
            raise ValueError(f"Context contains invalid character: {char}")
        context_indices.append(idx)
    
    # Convert indices to a tensor
      context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)  # Shape: (1, len(context))
      
    # Pass the tensor through the forward method of the model
      with torch.no_grad():  # Ensure no gradient computation for evaluation
        log_probs, _ = self.forward(context_tensor)
      # in forward of transformer layer: we have condition
      # if batch_size == 1, out = out.squeeze(0)
      # hence, log_probs.dim()==2 for all cases and log_probs.shape=(seq_length,vocab_size=27)
      if log_probs.shape[0]==0:
        raise ValueError(f"Model output is empty for the context : {context}")
      next_char_log_probs=log_probs[log_probs.shape[0]-1]
      # Return the log probabilities as a NumPy array
      return next_char_log_probs.cpu().numpy()
    
    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        self.eval()
        total_log_prob=0.0  
        if not context:  # Handle empty next_chars
        # Return the log probability of entire context/ this will be useful in
        # evaluating log-prob of language model
          current_context = " "        
          for j in range(len(next_chars)):
            total_log_prob+=self.get_next_char_log_probs(current_context)[self.vocab_index.index_of(next_chars[j])]
            current_context = current_context + next_chars[j]
          return total_log_prob

        else:  
          current_context=context
        # Iterate through each character in next_chars
          for i in range(len(next_chars)):
            total_log_prob+=self.get_next_char_log_probs(current_context)[self.vocab_index.index_of(next_chars[i])]
    # Construct the context for the current character
            current_context += next_chars[i]
          return total_log_prob  

    
    def generate_sequence(self, start_text, vocab_index, gen_length=40, num_pos=20):
      # Enforced: gen_length > num_pos=20 and len(start_text) < gen_length  
      # Initialize the generated text with the starting portion
      if len(start_text) >= gen_length: 
        raise ValueError("Start text length must be less than gen_length")
      
      generated_text = start_text

      
      while len(generated_text) <= num_pos-1:
        current_context = generated_text
        nxt_char = vocab_index.get_object(np.argmax(self.get_next_char_log_probs(current_context)))
        generated_text += nxt_char  
        
      while len(generated_text) <= gen_length -1 : 
        current_context = generated_text[-num_pos+1:]
        nxt_char = vocab_index.get_object(np.argmax(self.get_next_char_log_probs(current_context)))
        generated_text += nxt_char 
      return generated_text
    
    
def prepare_data(text, seq_length,vocab_index):
    input_data = []
    target_data = []
    for i in range(len(text) - seq_length):
      input_seq =[vocab_index.index_of(char) for char in text[i:i+seq_length]]
      target_seq = [vocab_index.index_of(char) for char in text[i+1:i+seq_length+1]]             
      input_data.append(input_seq)
      target_data.append(target_seq)
    return torch.tensor(input_data), torch.tensor(target_data)

"""
def beam_search_generate_sequence(lm, start_text, vocab_index, gen_length=40, num_pos=20, beam_size=10):

    if len(start_text) >= gen_length:
        raise ValueError("Start text length must be less than gen_length")

    # Initialize beams: start with the initial text and log probability of 0.0
    beams = [(start_text, 0.0)]  # (generated_text, cumulative_log_prob)
    for _ in range(gen_length-len(start_text)):
      new_beams=[]
      for text, log_prob_sum in beams:
        current_context=text[-num_pos:]
        log_probs = lm.get_next_char_log_probs(current_context)
        topk_indices = np.argsort(log_probs)[-beam_size:][::-1]
        topk_values = log_probs[topk_indices]
        for idx,log_prob in zip(topk_indices,topk_values):
          char = vocab_index.get_object(idx)  
          new_text = text + char 
          new_log_prob_sum = log_prob_sum + log_prob
          new_beams.append((new_text,new_log_prob_sum))
      beams = sorted(new_beams, key = lambda x: x[1], revers=True)[:beam_size] 

    return beams[0][0]      
"""      


def train_lm(args, train_text, dev_text, vocab_index,save_path):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Hyperparameters
    vocab_size = args.vocab_size
    num_positions = args.num_positions  # e.g., max sequence length
    d_model = args.d_model
    d_internal = args.d_internal
    num_classes=args.num_classes
    num_heads = args.num_heads
    num_layers = args.num_layers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    causalMask = args.causalMask
    model = TransformerLM(vocab_size,num_positions,d_model,d_internal,num_classes,num_heads,num_layers,causalMask,vocab_index)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Prepare data for training
    train_input, train_target = prepare_data(train_text, num_positions, vocab_index)
    dev_input, dev_target = prepare_data(dev_text, num_positions,vocab_index)
    train_dataset=TensorDataset(train_input,train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_input, batch_target in train_loader:
            optimizer.zero_grad()
            log_probs, _ = model(batch_input)  # Forward pass
            loss = criterion(log_probs.view(-1, vocab_size), batch_target.view(-1))  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        # Validation
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            dev_log_probs, _ = model(dev_input)
            dev_loss = criterion(dev_log_probs.view(-1, vocab_size), dev_target.view(-1))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader)}, Dev Loss: {dev_loss.item()}")
        model.train()  # Switch back to training mode for next epoch
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")    
    
    return model
    