import torch
from tqdm import tqdm

'''
The following are the shapes for everything
B = number of batches (i.e. number of sub-sequences)
I = input size (i.e. can be vocab size if one hot or embedding size if using word embedding)
BLength / S / sequence_length = the length of the single sub-sequence

H = embedding size the higher the more learnable parameters there is
Y / O / T = all the same but different names for contexts, it's basically output size which is always equal to vocab size

Wxh = (I, H) # weights to transform input token into a hidden_state
Whh = (H, H) # weights to turn the old hidden state into new hidden state (embedding the new token in the old memory)
Why = (H, Y) # weights to turn the new hidden state (containting old and new tokens) into output prediction for the next token

Example to understand batches: if your data is

data = [
    [0, 1, 3, 5, 6 , 7], # wikipedia article #1 tokenized in global token index (token2index dictionary used for tokenization)
    [3, 6, 8, 8] # second article and so on
]

data[0] is called a sequence
if you have batch size equal to 2 then X for the first sequence (article) is going to be:
X = 
[
    [0, 1, 3], # sub-sequence #1 of the sequence (article) 1, sorry for your brain bud, I had to look at 5 worse codes than this to understand and write it myself.
    [5, 6, 6]
]

then you take that last matrix and replace each element with the one-hot / embedding for it, turning this 2d matrix into 3d matrix,

basically imagine if there is # of (vocab_size / embedding size) layers of this matrix to represent the input as it's embedding
not as it's index ... got me ? I'm sure not, these stuff are brain-twisting, I'm not even writing this for you, I'm writing for me
to help myself keep my sanity by teaching what I'm learning.

hidden state will have the shape of (B, H)
'''


class RNN():
    def __init__(self, input_size, hidden_size, output_size,
                 token_to_index, index_to_token, index_to_embedding,
                 one_hot=True, eos="<eos>"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        scale = 0.01

        self.wxh = torch.rand(input_size, hidden_size, requires_grad=True).to(self.device)  * scale
        self.whh = torch.rand(hidden_size, hidden_size, requires_grad=True).to(self.device) * scale
        self.why = torch.rand(hidden_size, output_size, requires_grad=True).to(self.device) * scale

        self.bh = torch.rand(hidden_size, requires_grad=True).to(self.device)
        self.hy = torch.rand(output_size, requires_grad=True).to(self.device)

        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
        self.index_to_embedding = index_to_embedding

        self.one_hot = one_hot
        self.eos = self.token_to_index[eos]

    def embed(self, index, one_hot=None):
        if one_hot is None: 
            one_hot = self.one_hot

        if one_hot:
            vector = torch.zeros(self.input_size, device=self.device)
            vector[index] = 1.0
            return vector

        return self.index_to_embedding[index].to(self.device)

    def forward(self, x_t, hidden_state):
        new_hidden_state = torch.tanh(
            x_t @ self.wxh + 
            hidden_state @ self.whh    +
            self.bh
        ).to(self.device)

        output_logits = (new_hidden_state @ self.why + self.hy).to(self.device)

        return output_logits, new_hidden_state
    
    # TODO: gpu the heck out of it
    def sequence_to_hidden(self, sequence, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.get_fresh_hidden_state().to(self.device)

        with torch.no_grad():
            for token_index in sequence:
                token = self.embed(token_index).to(self.device)
                output, hidden_state = self.forward(token, hidden_state)
            
            return hidden_state

    def get_fresh_hidden_state(self, n_batches):
        return torch.zeros(n_batches, self.hidden_size, requires_grad=True).to(self.device)
    
    def sample(self, sequence, max_length=2000, deterministic=False):
        generated = list(sequence)
        hidden_state = self.sequence_to_hidden(generated)
        
        for _ in range(max_length):
            with torch.no_grad():
                last_idx = generated[-1]
                token = self.embed(last_idx)
                output_logits, hidden_state = self.forward(token, hidden_state)
                probabilities = torch.softmax(output_logits, dim=0)
                
                if deterministic:
                    output_idx = torch.argmax(probabilities).item()
                else:
                    output_idx = torch.multinomial(probabilities, 1).item()
                
                if output_idx == self.eos_index:
                    break

                generated.append(output_idx)
                

        output_sequence = [self.index_to_token[idx] for idx in generated]
        return "".join(output_sequence)
        
    
    def train(self, sequences, epochs, checkpoints=[], batch_size=1, learning_rate=0.01, max_grad_norm=5.0):
        for epoch in tqdm(range(epochs), desc="Epoch", position=0):
            epoch_loss = 0
            
            for sequence in tqdm(sequences, desc="sequence", position=1, leave=False):
                
                hidden_state = self.get_fresh_hidden_state(n_batches=batch_size)
                
                sequence_loss = 0

                subsequence_length = len(20)
                subsequences_start_indexes = range(0, len(sequence) - subsequence_length + 1, subsequence_length)
                subsequences = [sequence[i][i + subsequence_length] for i in subsequences_start_indexes]
                for sequence_index, subsequence in enumerate(subsequences):
                    X = torch.zeros(batch_size, subsequence_length, self.input_size, device=self.device)

                    for token_index in range(subsequence_length):
                        
                        # current_token_index_in_subsequence = sub_sequences_start_index + token_index
                        # next_token_index_in_subsequence = current_token_index_in_subsequence + 1

                        # if next_token_index_in_subsequence >= len(sequence):
                        #     break

                        # current_token_global_onehot_index = sequence[current_token_index_in_subsequence]
                        # next_token_global_onehot_index = sequence[next_token_index_in_subsequence]

                        X[sequence_index, token_index, token] = 1.0
                        

                        for t in range(subsequence_length):
                            # ()
                            x_t = X[t]
                            
                            # (B, I) ,,,,,, (B, H)
                            output_logits, hidden_state = self.forward(x_t, hidden_state)
                            
                            # (B, I)
                            probabilities = torch.softmax(output_logits, dim=1)

                            loss = torch.mean(
                                -torch.log(
                                    probabilities[] + 1e-9
                                )
                            )



                    # Forward pass through entire sequence
                    for i in range(len(sequence) - 1):
                        token = self.embed(sequence[i])
                        output_logits, hidden_state = self.forward(token, hidden_state)
                        probabilities = torch.softmax(output_logits, dim=0)
                        target_index = sequence[i + 1]
                        
                        # Accumulate loss
                        sequence_loss += -torch.log(probabilities[target_index] + 1e-9)

                epoch_loss += sequence_loss.item()

            # Epoch statistics
            avg_loss = epoch_loss / len(sequences)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if (epoch + 1) in checkpoints:
                print(f"checkpoint at epoch {epoch + 1}")
                checkpoints.append(self)
        
        return checkpoints.append(self)
