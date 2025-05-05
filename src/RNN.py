import torch
from tqdm import tqdm

class RNN():
    def __init__(self, input_size, hidden_size, output_size,
                 token_to_index, index_to_token, index_to_embedding,
                 one_hot=True, eos="<eos>"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.wxh = torch.rand(hidden_size, input_size, requires_grad=True).to(self.device)  * 0.01
        self.whh = torch.rand(hidden_size, hidden_size, requires_grad=True).to(self.device) * 0.01
        self.why = torch.rand(output_size, hidden_size, requires_grad=True).to(self.device) * 0.01

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

    def forward(self, input_embedding, hidden_state):
        new_hidden_state = torch.tanh(
            self.wxh @ input_embedding + 
            self.whh @ hidden_state    +
            self.bh
        ).to(self.device)

        output_logits = (self.why @ new_hidden_state + self.hy).to(self.device)

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

    def get_fresh_hidden_state(self):
        return torch.zeros(self.hidden_size, requires_grad=True).to(self.device)
    
    def sample(self, sequence, max_length=2000, deterministic=False):
    # Use copy to avoid modifying input
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
        
    
    def train(self, sequences, epochs, checkpoints=[], learning_rate=0.01, max_grad_norm=5.0):
        for epoch in tqdm(range(epochs), desc="Epoch", position=0):
            epoch_loss = 0
            
            for sequence in tqdm(sequences, desc="sequence", position=1, leave=False):
                hidden_state = self.get_fresh_hidden_state()
                sequence_loss = 0

                # Forward pass through entire sequence
                for i in range(len(sequence) - 1):
                    token = self.embed(sequence[i])
                    output_logits, hidden_state = self.forward(token, hidden_state)
                    probabilities = torch.softmax(output_logits, dim=0)
                    target_index = sequence[i + 1]
                    
                    # Accumulate loss
                    sequence_loss += -torch.log(probabilities[target_index] + 1e-9)

                # Calculate gradients after full sequence
                params = [self.wxh, self.whh, self.why, self.bh, self.hy]
                grads = torch.autograd.grad(sequence_loss, params, retain_graph=False)

                # Global gradient clipping
                total_norm = torch.sqrt(sum(torch.sum(grad**2) for grad in grads if grad is not None))
                if total_norm > max_grad_norm:
                    clip_coef = max_grad_norm / (total_norm + 1e-6)
                    grads = [grad * clip_coef if grad is not None else None for grad in grads]

                # Parameter update
                with torch.no_grad():
                    for param, grad in zip(params, grads):
                        if grad is not None:
                            param -= learning_rate * grad

                epoch_loss += sequence_loss.item()

            # Epoch statistics
            avg_loss = epoch_loss / len(sequences)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if (epoch + 1) in checkpoints:
                print(f"checkpoint at epoch {epoch + 1}")
                checkpoints.append(self)
        
        return checkpoints.append(self)
