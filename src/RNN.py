import torch
from tqdm import tqdm
import copy

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

        scale_of_weights_init = 0.01

        # self.wxh = torch.rand(input_size, hidden_size, requires_grad=True , device=self.device) * scale_of_weights_init
        # self.whh = torch.rand(hidden_size, hidden_size, requires_grad=True, device=self.device) * scale_of_weights_init
        # self.who = torch.rand(hidden_size, output_size, requires_grad=True, device=self.device) * scale_of_weights_init

        self.wxh = torch.rand(input_size, hidden_size, requires_grad=True , device=self.device)
        self.whh = torch.rand(hidden_size, hidden_size, requires_grad=True, device=self.device)
        self.who = torch.rand(hidden_size, output_size, requires_grad=True, device=self.device)


        self.bh = torch.rand(hidden_size, requires_grad=True, device=self.device)
        self.bo = torch.rand(output_size, requires_grad=True, device=self.device)

        self.index_to_token = index_to_token
        self.token_to_index = token_to_index
        self.index_to_embedding = index_to_embedding

        self.one_hot = one_hot
        self.eos = self.token_to_index[eos]

        self.learning_rate = torch.tensor(0.001)
        self.parameters = [self.wxh, self.whh, self.who, self.bh, self.bo]
        self.optimizer = torch.optim.AdamW(self.parameters, self.learning_rate)

    def embed(self, index, one_hot=None):
        if one_hot is None:
            one_hot = True
        
        if one_hot:
            embedding = torch.zeros(self.input_size, device=self.device)
            embedding[index] = 1.0
            return embedding

        return torch.tensor(self.index_to_embedding[index], device=self.device)
            

    def forward(self, x_t, hidden_state):
        # (B, H)
        new_hidden_state = torch.tanh(
            x_t @ self.wxh +
            hidden_state @ self.whh + 
            self.bh
        )

        # (B, O)
        output = new_hidden_state @ self.who + self.bo

        return new_hidden_state, output

    def sequence_to_hidden(self, sequence, hidden_state=None):
        with torch.no_grad():
            if hidden_state is None:
                hidden_state = self.get_fresh_hidden_state(batch_size=1)

            for i in range(len(sequence)):
                # (B, I)
                x = torch.zeros(1, self.input_size)
                
                x[0] = self.embed(sequence[i])

                # (B, H)      (B, O)
                hidden_state, output = self.forward(x, hidden_state)

            return hidden_state
            
    
    def get_fresh_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=self.device)
    
    def generate(self, sequence, max_length=100, deterministic=False):
        with torch.no_grad():
            generated_sequence = []

            hidden_state = self.get_fresh_hidden_state(batch_size=1)
            hidden_state = self.sequence_to_hidden(sequence, hidden_state)

            last_token_index = sequence[-1]

            for i in tqdm(range(max_length), desc="Tokens Generated:"):
                # (B, I)
                x = torch.zeros(1, self.input_size)
                
                x[0] = self.embed(last_token_index)

                # (B, H)      (B, O)
                hidden_state, output = self.forward(x, hidden_state)

                probabilities = torch.softmax(output[0])

                output_token_index = torch.argmax(probabilities).item()

                if not deterministic:
                    output_token_index = torch.multinomial(probabilities, num_samples=1).item()

                generated_sequence.append(output_token_index)

                last_token_index = output_token_index

                if last_token_index == self.eos:
                    break
            
            return ''.join([self.index_to_token[index] for index in generated_sequence])
        
    
    def train(self, sequences, epochs, checkpoints=[], batch_size=1, sub_sequence_length=None, max_grad_norm=5.0):
        torch.autograd.set_detect_anomaly(True)
        models = []
        for epoch in tqdm(range(epochs), desc="Epochs Finished: ", position=0):
            epoch_loss = 0
            for sequence in tqdm(sequences, desc="Sequences Finished:", position=1, leave=False):
                # I swear these comments and code are written by a human (me), no AI involved here.
                
                # --- getting sub-sequences ready for training ---
                # sequence is like [0, 52, 34, 60]
                # sub-sequences will be like (batch size equal 2 which is how many rows we have):
                    # [
                    #   [0, 52],
                    #   [34, 60],
                    #] ... so if sub_sequence_length is not explicitly provided, we just infer it, if it's explicitly provided then we use it and do many batches till the sub-sequences are all finished
                
                # TODO: Prime factorization if batch_size is None to use as much of the training data as possible

                if sub_sequence_length is None:
                    if batch_size > len(sequence):
                        sub_sequence_length = len(sequence) # use full
                    else:
                        sub_sequence_length = len(sequence) // batch_size

                if sub_sequence_length < 2:
                    continue

                # the point of the following is to create subsequences with the previously defined length but to ensure they all have the same size, ignore the reminder of the main sequence if you have to
                # if sequence is [34, 66, 77, 99, 55] and length is 2 then: 
                # stop_range = len(sequence) - sub_sequence_length # this will be 5 - 2 = 3
                # sub_sequences_start_indexes = list(range(0, stop_range, sub_sequence_length)) # this will be [0, 2]
                
                # TODO: this can be optimized for memroy and time but I'm trying to prevent premature optimization
                # sub_sequences = [sequence[start_index: start_index + sub_sequence_length] for start_index in sub_sequences_start_indexes]
                # this will be [[34, 66], [77, 99]]
                # notice how the element 55 was discarded! because the sequence doesn't have enough tokens.

                num_sub_sequences_to_form = len(sequence) // sub_sequence_length
                
                if num_sub_sequences_to_form == 0:
                    # Sequence is too short to form any sub-sequence of the required length.
                    sub_sequences = []
                else:
                    sub_sequences = [
                        sequence[i * sub_sequence_length : (i + 1) * sub_sequence_length]
                        for i in range(num_sub_sequences_to_form)
                    ]

                M = len(sub_sequences)

                if M == 0:
                    continue

                # --- hidden state init ---
                # (B, H) shape, basically an (H) shape vector for every sub_sequence :)
                hidden_state = self.get_fresh_hidden_state(batch_size=M)

                

                # --- getting training X and Y ready ---
                # Recall how batch_size is simply how many sub-sequences to train at once before calculating loss and gradients and so on

                # every column is a subsequence of elements like [33, 66], but because 33 needs to be turned into onehot / embedding, we have to account for that by shape I
                # every row is a time t of the RNN, so we roll it row by row .. for example:

                '''
                    x = 
                    [
                        [embedding_of_34, embedding_of_77],
                        [embedding_of_66, embedding_of_99]
                    ]
                    
                    notice that the embedding of an element can be it's onehot or it's embedding vector depending on your choice / context (and in both cases that embedding is also a vector, hence 3d matrix)
                    also notice that we process this row by row, where each column has the current element of that column's sub-sequence
                    we have as many columns as batch size because that's how many subsequences we have
                    we have as many rows as sub-sequence's length because that's how many stpes we have in the training which is also the number of sub-sequence length
                '''
                # (S, B, I)
                X = torch.zeros(sub_sequence_length, M, self.input_size, device=self.device)

                '''
                TODO: fix this description
                    Y is an output onehot vector representing logits of next token prediction ... one for every subsequence we are working with
                    note that every column in Y will have only 1 element with value 1, the rest will be zero because it's a onehot encoding for the ground truth of the prediction
                    might seem weird but you'll understand when you see the loss hehe :)

                    also note that we could have made it of shape (S, B, O) but will make things more complicated for no reason, just keep going in the code and you'll get it :)
                '''
                Y = torch.zeros(sub_sequence_length, M, self.output_size, device=self.device)

                # populating x, y with training data
                for sub_sequence_index, sub_sequence in enumerate(sub_sequences):
                    for time_step, element in enumerate(sub_sequence):
                        X[time_step, sub_sequence_index] = self.embed(element)
                        
                        if time_step != 0:
                            Y[time_step - 1][sub_sequence_index] = self.embed(element, one_hot=True)


                # --- training steps ---
                for i in range(sub_sequence_length-1): # recall that the last step in the subsequence has no ground truth (no next token!) so we don't train on it, TODO: use different Y technique to not waste training data
                    # (B, H)
                    x_t = X[i] # current time step input, i.e. i-th token in the sub-sequnce.... for every subsequence, shape (B, I), example: [embedding_of_34, embedding_of_77]

                    # making a prediction for all sub-sequences ... all at once! and upadating hidden state for all subsequences
                    # (B, H)      (B, O)
                    hidden_state, output = self.forward(x_t, hidden_state=hidden_state)
                    
                    # Detach hidden_state to prevent graph reuse
                    hidden_state = hidden_state.detach()

                    # every column of output represnets the logits of prediciion for the subsequence at that column, we softmax these logits
                    # (B, O) shape didn't change, only each column is softmaxed
                    softmax_output = torch.softmax(output, dim=1)

                    loss = torch.mean(
                        # (B, 1)
                        -torch.log(   #(B, H)  (B, H)
                            torch.sum(softmax_output * Y[i], dim=1, keepdim=True) + 1e-9 # notice this is pair-wise multiplication, not matrix dot product, 1e-9 for numerical stability
                        )
                    )

                    epoch_loss += loss.item()

                    self.optimizer.zero_grad()

                    loss.backward()

                    # avoiding exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=max_grad_norm)

                    # for param in self.parameters:
                    #     if param.grad is not None:
                    #         torch.nn.utils.clip_grad_norm_([param], max_norm=max_grad_norm)

                    self.optimizer.step()
            
            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(sequences)}")
            if epoch + 1 in checkpoints:
                models.append(copy.deepcopy(self))
                print(f"Model Saved @ Epoch {epoch + 1}")
        
        models.append(copy.deepcopy(self))
        return models