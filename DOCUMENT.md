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

This matters for large scores: `exp(100)` overflows float32 (max ≈ 3.4e38), so the naive version computes `inf / inf = nan`:

```python
large_scores = torch.tensor([100.0, 101.0, 102.0])
softmax_naive(large_scores)               # tensor([nan, nan, nan])
torch.softmax(large_scores, dim=-1)       # tensor([0.0900, 0.2447, 0.6652])
```

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

## Self-attention with trainable weights

Self-attention adds 3 trainable weight matrices $W_q$, $W_k$, $W_v$ to the simplified attention above. Each input $x^{(i)}$ is projected into a query, a key and a value vector:

$$q^{(i)} = x^{(i)} W_q, \qquad k^{(i)} = x^{(i)} W_k, \qquad v^{(i)} = x^{(i)} W_v$$

The attention scores now come from queries and keys instead of the raw inputs, and the context vector is a weighted sum of the values.

PyTorch implementation for query 2 ("journey"):

```python
d_in = inputs.shape[1]  # input embedding size, 3
d_out = 2               # q/k/v size, kept different from d_in to make the shapes visible

torch.manual_seed(123)
W_query = nn.Parameter(torch.rand(d_in, d_out))
W_key = nn.Parameter(torch.rand(d_in, d_out))
W_value = nn.Parameter(torch.rand(d_in, d_out))

x_2 = inputs[1]
query_2 = torch.matmul(x_2, W_query)    # (d_out,)
keys = torch.matmul(inputs, W_key)      # (T, d_out)
values = torch.matmul(inputs, W_value)  # (T, d_out)
```

## Scaled dot-product attention

The scores are divided by $\sqrt{d_k}$ (the key size) before softmax:

$$\alpha_{2j} = \text{softmax}\left(\frac{q^{(2)} \cdot k^{(j)}}{\sqrt{d_k}}\right), \qquad z^{(2)} = \sum_{j=1}^{T} \alpha_{2j}\, v^{(j)}$$

PyTorch implementation:

```python
attn_scores_2 = torch.matmul(query_2, keys.T)  # (T,)
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = torch.matmul(attn_weights_2, values)  # (d_out,)
```

### Why scale by $\sqrt{d_k}$?

Dividing by $\sqrt{d_k}$ stops the scores from growing with the vector size. Without it, softmax pushes almost all the weight onto one or two tokens, and training stalls.

**Why the scores grow.** A dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ adds up $d_k$ terms. If each $q_i$ and $k_i$ is roughly zero-mean with variance 1, the sum has variance $d_k$, so a typical score is about $\sqrt{d_k}$ in size. Dividing by $\sqrt{d_k}$ brings the scores back to variance about 1, whatever the head size.

**What large scores do to softmax.** Because of the exponential, large differences between scores send almost all the weight to the largest one. With random $q$ and $k$ for 6 tokens:

| $d_k$ | Version | Score std | Attention weights |
|---|---|---|---|
| 2 | unscaled | 1.27 | `[0.011, 0.105, 0.537, 0.137, 0.139, 0.071]` |
| 2 | scaled | 0.90 | `[0.027, 0.132, 0.419, 0.160, 0.162, 0.101]` |
| 128 | unscaled | 7.41 | `[0.016, 0.000, 0.580, 0.000, 0.000, 0.403]` |
| 128 | scaled | 0.65 | `[0.192, 0.129, 0.263, 0.114, 0.046, 0.255]` |

At $d_k = 2$ scaling barely matters. At $d_k = 128$, a typical head size in real models, the unscaled version gives three tokens zero weight.

**Why that stalls training.** The gradient of softmax for token $j$ is proportional to its weight $\alpha_j$. Tokens with weight ≈ 0 get almost no gradient, so $W_q$ and $W_k$ get no signal to learn whether those tokens should have been attended to.

**Why the simplified version didn't scale.** It had no trainable weights, so there were no gradients to protect, and its vectors were only 3 long, so the scores stayed small. The scaling comes from the original Transformer paper, which is why this is called *scaled* dot-product attention.

## Self-attention for all inputs

With $X$ holding the inputs as rows:

$$Q = X W_q, \quad K = X W_k, \quad V = X W_v, \qquad Z = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Wrapping the three matrices in an `nn.Module` registers them as trainable parameters.

PyTorch implementation:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = torch.matmul(torch.matmul(x, self.W_query), torch.matmul(x, self.W_key).T)  # (T, T)
        attn_weights = torch.softmax(attn_scores / self.W_key.shape[-1]**0.5, dim=-1)
        return torch.matmul(attn_weights, torch.matmul(x, self.W_value))                         # (T, d_out)

torch.manual_seed(123)  # same seed as above, so the module gets the same three matrices
context_vecs = SelfAttention(d_in, d_out)(inputs)
```

Row 2 of `context_vecs` matches `context_vec_2` from above.

## Efficient self-attention

Two changes over `SelfAttention`:

1. **One projection instead of three.** $W_q$, $W_k$ and $W_v$ are stored side by side as one $(d_{in}, 3\,d_{out})$ matrix, so a single matmul produces $Q$, $K$ and $V$.
2. **Fused attention.** `F.scaled_dot_product_attention` does the scaling, softmax and weighted sum in one kernel. On NVIDIA GPUs it uses FlashAttention, which is faster and never stores the full $(T, T)$ weight matrix. On a Mac CPU or MPS it runs at about the same speed as the unfused version.

PyTorch implementation:

```python
import torch.nn.functional as F

class SelfAttentionBest(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # W_q, W_k, W_v side by side; created in the same order as SelfAttention so the same seed gives the same values
        self.W_qkv = nn.Parameter(torch.cat([torch.rand(d_in, d_out) for _ in range(3)], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries, keys, values = torch.matmul(x, self.W_qkv).chunk(3, dim=-1)  # each (T, d_out)
        return F.scaled_dot_product_attention(queries, keys, values)          # (T, d_out)

torch.manual_seed(123)
context_vecs_best = SelfAttentionBest(d_in, d_out)(inputs)
```

`context_vecs_best` matches `context_vecs` from `SelfAttention` (within `atol=1e-6`).
