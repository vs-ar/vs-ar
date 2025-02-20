### **How BERT Handles Batch Size: Mathematical & Computational Explanation**
BERT (**Bidirectional Encoder Representations from Transformers**) processes text **in parallel** rather than sequentially (like LSTMs). It relies on **self-attention** and **fully connected layers**, making it highly efficient for batch processing.

---
## **1. Given Input Shape (Batch Size, Sequence Length, Embedding Size)**
Let’s assume a **BERT input of size**:
\[
(4, 128, 1266)
\]
where:
- **Batch Size (B) = 4** → 4 sentences (or text sequences) are processed at once.
- **Sequence Length (T) = 128** → Each sequence has **128 tokens**.
- **Embedding Size (D) = 1266** → Each token is represented by a **1266-dimensional vector**.

---
## **2. BERT Encoder Block (Mathematical Computation)**
Each **BERT layer** consists of:
1. **Multi-Head Self-Attention**
2. **Feedforward Network (FFN)**
3. **Layer Normalization & Residual Connections**

Each **input tensor** \( X \) goes through these operations in parallel across all batches.

---
## **3. Multi-Head Self-Attention (MHA)**
BERT computes self-attention using:
\[
A = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
\]
where:
- \( Q = X W_Q \), \( K = X W_K \), \( V = X W_V \)
- \( W_Q, W_K, W_V \) are **weight matrices** of size **(D, d_k)**
- \( d_k \) is the **dimension of each attention head**
- \( A \) is the **attention score matrix**.

### **Matrix Multiplications for Batches**
#### **Step 1: Compute Query, Key, Value Matrices**
\[
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
\]
- \( X \in \mathbb{R}^{(4, 128, 1266)} \)
- \( W_Q, W_K, W_V \in \mathbb{R}^{(1266, d_k)} \)

#### **Step 2: Compute Attention Scores**
\[
\frac{QK^T}{\sqrt{d_k}}
\]
- \( Q \) and \( K \) have shape **(4, 128, d_k)**
- Transpose \( K \) to **(4, d_k, 128)**
- Multiplication:  
  \[
  (4, 128, d_k) \times (4, d_k, 128) = (4, 128, 128)
  \]

#### **Step 3: Apply Softmax and Multiply by \( V \)**
\[
A V
\]
- Attention score matrix **(4, 128, 128)**
- \( V \) has shape **(4, 128, d_k)**
- Final multiplication:
  \[
  (4, 128, 128) \times (4, 128, d_k) = (4, 128, d_k)
  \]

Each batch sequence gets processed **independently**, leveraging **parallel matrix multiplications**.

---
## **4. Feedforward Network (FFN)**
Each token undergoes a **position-wise feedforward operation**:

\[
\text{FFN}(X) = \max(0, X W_1 + b_1) W_2 + b_2
\]
where:
- \( W_1 \in \mathbb{R}^{(D, d_{ff})} \), \( W_2 \in \mathbb{R}^{(d_{ff}, D)} \)
- Biases: \( b_1 \in \mathbb{R}^{(d_{ff})} \), \( b_2 \in \mathbb{R}^{(D)} \)

For a batch:
\[
(4, 128, 1266) \times (1266, d_{ff}) = (4, 128, d_{ff})
\]
\[
(4, 128, d_{ff}) \times (d_{ff}, 1266) = (4, 128, 1266)
\]

---
## **5. How BERT Handles Batch Size Efficiently**
1. **Self-attention is Fully Parallelized**  
   - Unlike LSTMs, BERT **does not process tokens sequentially**.
   - Each **batch** is processed **simultaneously** using **matrix multiplications**.

2. **Broadcasting for Bias Addition**  
   - **Bias terms** are broadcasted across **batch & sequence dimensions**.

3. **Batch Computation in PyTorch**  
   - PyTorch's implementation of `torch.nn.Linear()` and `torch.nn.MultiheadAttention()` internally handles **batch operations**, making matrix multiplications extremely fast.

---
## **6. PyTorch Code for Batched BERT Computation**
```python
import torch
from transformers import BertModel, BertTokenizer

# Load Pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Example batch of 4 sentences
sentences = [
    "The cat sat on the mat.",
    "I love natural language processing.",
    "BERT is a powerful transformer model.",
    "How does batch processing work in BERT?"
]

# Tokenize input
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)

# Forward Pass (Batch Processing)
with torch.no_grad():
    outputs = model(**inputs)

# Print output shapes
print("Token Embeddings Shape:", outputs.last_hidden_state.shape)  # (4, 128, 768)
print("CLS Embedding Shape:", outputs.pooler_output.shape)  # (4, 768)
```

---
## **7. Summary**
### **How BERT Handles Batch Processing**
| Feature | **BERT (Transformer)** | **LSTM (Recurrent)**
|---------|-----------------|----------------|
| Processing | **Parallel (O(1))** | **Sequential (O(T))** |
| Batch Computation | Fully **vectorized** | **Iterative** |
| Dependencies | **Global** (All tokens interact) | **Local** (Only past tokens) |
| Memory | **Higher (Multi-Head Attention)** | **Lower (Fewer weights)** |

### **Key BERT Computation Steps**
1. **Token Embeddings Computed in Parallel**  
   - Shape: \( (B, T, D) = (4, 128, 1266) \)

2. **Multi-Head Self-Attention (MHA)**  
   - Computes **batch matrix multiplications**:
     \[
     A = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
     \]
   - Shape: \( (4, 128, d_k) \)

3. **Feedforward Layer (FFN) Applied in Parallel**  
   - Uses matrix multiplications:  
     \[
     (4, 128, 1266) \times (1266, d_{ff}) = (4, 128, d_{ff})
     \]

4. **Final Hidden States Output**  
   - Shape: \( (4, 128, 768) \) for **BERT-Base**.

---
## **8. Key Differences Between LSTM and BERT in Batch Processing**
| **Feature** | **LSTM** | **BERT** |
|------------|---------|----------|
| **Processing Type** | **Sequential** | **Fully Parallel** |
| **Batch Computation** | Slow (Iterative) | Fast (Matrix Multiplication) |
| **Dependency Handling** | Uses **hidden state** | Uses **self-attention** |
| **Performance** | Struggles with **long-range dependencies** | **Handles long-range dependencies easily** |
| **Output Shape** | **(Batch, Time, Hidden Size)** | **(Batch, Time, Embedding Size)** |

---

## **Conclusion**
- **BERT handles batch processing efficiently** by **vectorized matrix multiplications** rather than sequential updates.
- **Self-attention enables all tokens to be processed in parallel**, making it **faster than LSTMs**.
- **BERT is ideal for large-scale NLP tasks**, while **LSTMs are more suited for small datasets**.

This explains **how BERT processes batches mathematically and computationally**, just like we did for LSTMs. Let me know if you need further clarifications!
