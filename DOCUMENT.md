# Notes

## Goal of Simplified Self Attention

Calculate a context vector $Z^{(i)}$ for each element $x^{(i)}$

![Figure 1: Attention scores for the query "journey"](./document_screenshots/query_self_attention.png)

**Figure 1:** Attention scores $\omega_{2j}$ between the query $x^{(2)}$ ("journey") and each input $x^{(j)}$. *Image credit: Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Manning, 2024.*

The First task is to extract the intermediate value `w` which is also called the attention scores (see [Figure 1](#figure-1)).

For a query an a set of input embeddings we have to quantify how much attention we should pay to each of these other embeddings. Here the query is coming from one the the words in the input sequence thus self attention. In [Figure 1](#figure-1) we are queriying for the word "journey".

$w_{2 \leftarrow 1}$ or $w_{21}$ is the attention assigned to word 1 for query 2.

$W_2$ = $x_2$ * $X^T$
where $X^T$ have all the embeddings vectors as columns.

## Normalizing attention scores with softmax

![Figure 2: Normalizing attention scores into attention weights](./document_screenshots/attention_weights_normalization.png)

**Figure 2:** The attention scores $\omega_{2j}$ for the query $x^{(2)}$ are normalized into attention weights $\alpha_{2j}$. *Image credit: Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Manning, 2024.*

The attention scores are normalized into attention weights $\alpha_{2j}$ that are positive and sum to 1 (see [Figure 2](#figure-2)). Softmax is preferred over dividing by the plain sum because it keeps every weight positive and gives more weight to larger scores:

$$\alpha_{2j} = \text{softmax}(w_{2j}) = \frac{e^{w_{2j}}}{\sum_{k=1}^{T} e^{w_{2k}}}$$

PyTorch implementation:

```python
def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)

attn_weights_2 = softmax_naive(attn_scores_2)

# Built-in version: subtracts max(x) before exp() so large scores don't overflow
attn_weights_2 = torch.softmax(attn_scores_2, dim=-1)
```

`softmax_naive` computes the formula above directly. `torch.softmax` computes a numerically stable version that subtracts the largest score $m$ first:

$$m = \max_{k} w_{2k}$$

$$\alpha_{2j} = \frac{e^{w_{2j} - m}}{\sum_{k=1}^{T} e^{w_{2k} - m}}$$

Both give the same result, because the extra $e^{-m}$ factor cancels between numerator and denominator:

$$\frac{e^{w_{2j} - m}}{\sum_{k} e^{w_{2k} - m}} = \frac{e^{-m}\, e^{w_{2j}}}{e^{-m} \sum_{k} e^{w_{2k}}} = \frac{e^{w_{2j}}}{\sum_{k} e^{w_{2k}}}$$

After subtracting $m$, every exponent is $\le 0$, so each $e^{(\cdot)}$ lies in $(0, 1]$ and can't overflow.

## Computing the context vector

![Figure 3: Computing the context vector z(2)](./document_screenshots/context_vector.png)

**Figure 3:** Each input vector $x^{(j)}$ is multiplied by its attention weight $\alpha_{2j}$ and the results are summed to get the context vector $z^{(2)}$. *Image credit: Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Manning, 2024.*

The context vector is the weighted sum of all the input vectors (see [Figure 3](#figure-3)):

$$z^{(2)} = \sum_{j=1}^{T} \alpha_{2j}\, x^{(j)}$$

PyTorch implementation:

```python
query = inputs[1]  # "journey"
attn_scores_2 = torch.matmul(inputs, query)
attn_weights_2 = torch.softmax(attn_scores_2, dim=-1)

context_vec_2 = torch.zeros(query.shape)
for j, x_j in enumerate(inputs):
    context_vec_2 += attn_weights_2[j] * x_j

# Same weighted sum as a single matrix-vector product
context_vec_2 = torch.matmul(attn_weights_2, inputs)  # tensor([0.4419, 0.6515, 0.5683])
```

## Attention weights for all inputs

![Figure 4: Attention weight matrix for all inputs](./document_screenshots/attention_weight_matrix.png)

**Figure 4:** The attention weight matrix. Row $i$ holds the attention weights $\alpha_{ij}$ for query $x^{(i)}$, and each row sums to 1. The highlighted "journey" row is the one computed above. *Image credit: Sebastian Raschka, [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch), Manning, 2024.*

Repeating the same steps for every input as the query gives the full attention weight matrix (see [Figure 4](#figure-4)). With $X$ holding the input embeddings as rows:

$$W = X X^T, \qquad A = \text{softmax}(W) \text{ applied to each row}, \qquad Z = A X$$

Row $i$ of $Z$ is the context vector $z^{(i)}$.

PyTorch implementation:

```python
def simplified_self_attention(x: torch.Tensor) -> torch.Tensor:
    attn_scores = torch.matmul(x, x.T)                 # (T, T)
    attn_weights = torch.softmax(attn_scores, dim=-1)  # normalize each row
    return torch.matmul(attn_weights, x)               # (T, d)

context_vecs = simplified_self_attention(inputs)
```

`dim=-1` normalizes along the last dimension, so each row (one query) sums to 1. Row 2 of `context_vecs` matches `context_vec_2` from above.
