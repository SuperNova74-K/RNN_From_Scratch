## RNN Shape Definitions and Batching Explanation

This document outlines the common shapes and terminology used in Recurrent Neural Network (RNN) implementations, particularly focusing on how data is structured and processed in batches.

### Symbol Definitions

*   **B**: Number of batches (i.e., number of sub-sequences processed in parallel).
*   **I**: Input size. This can be:
    *   Vocabulary size if using one-hot encoding for input tokens.
    *   Embedding size if using word embeddings.
*   **S**: The length of a single sub-sequence within a batch.
*   **H**: Hidden state size (embedding size). A larger `H` means more learnable parameters in the model.
*   **O**: Output size. These symbols are often used interchangeably and typically represent the vocabulary size, as the output is usually a probability distribution over the vocabulary for the next token.

### Weight Matrix Shapes and Purposes

*   **Wxh**: Shape `(I, H)`
    *   These are the weights used to transform an input token (of size `I`) into a hidden state representation (of size `H`).
*   **Whh**: Shape `(H, H)`
    *   These weights are applied to the previous hidden state (`H`) to incorporate it into the calculation of the new hidden state (`H`). This allows the network to "remember" past information.
*   **Who**: Shape `(H, O)`
    *   These weights transform the new hidden state (which combines information from the current input and the previous hidden state) into an output prediction (of size `O`, typically the vocabulary size).

### Biases
*   **Bh**: Shape `(H)`
    *   Bias vector for input to hidden_state
*   **Bo**: Shape `(O)`
    *   Bias vector for hidden_state to output

### Example: Understanding Batches

Consider a dataset composed of tokenized sequences (e.g., articles):
