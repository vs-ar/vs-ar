# **LoRA (Low-Rank Adaptation) Explained: Forward & Backpropagation**
LoRA (**Low-Rank Adaptation**) is a technique for **efficient fine-tuning of large pre-trained models**, such as **BERT, GPT, LLaMA, etc.** Instead of updating all **dense weight matrices**, LoRA **adds low-rank trainable matrices** while **freezing the original weights**.

---

## **1. Why LoRA?**
Fine-tuning large models is **memory-intensive** because:
- It requires **storing full gradients** for each parameter.
- **Updating all weights** makes training slow and costly.

### **LoRA solves this by:**
- **Freezing the original weights** \( W \).
- **Introducing small trainable matrices** \( A \) and \( B \) with **low-rank \( r \) (e.g., 8 or 16)**.
- **Only fine-tuning these new parameters**, reducing GPU memory usage.

---

## **2. Forward Pass in LoRA**
### **LoRA Applied to a Linear Layer**
A standard **Transformer linear layer**:
\[
Y = XW
\]
where:
- \( X \in \mathbb{R}^{B \times T \times D} \) (**input activations**).
- \( W \in \mathbb{R}^{D \times H} \) (**pre-trained weight matrix**).
- \( Y \in \mathbb{R}^{B \times T \times H} \) (**output activations**).

### **LoRA Modification**
Instead of **fine-tuning \( W \)** directly, LoRA **adds** a **low-rank decomposition**:
\[
\tilde{W} = W + \Delta W
\]
where:
\[
\Delta W = A B
\]
and:
- \( A \in \mathbb{R}^{D \times r} \) (small matrix).
- \( B \in \mathbb{R}^{r \times H} \) (small matrix).

Thus, the forward pass becomes:
\[
Y = X(W + AB)
\]
\[
Y = XW + XAB
\]

### **What’s Happening?**
- **\( W \) is frozen**, so it is **not updated**.
- **\( A \) and \( B \) are trainable**, so **only these get updated**.
- **Memory savings**: Instead of training a full **\( D \times H \)** matrix, we only train **\( D \times r + r \times H \)** parameters.

---

## **3. Backpropagation in LoRA**
During backpropagation, we need to compute **gradients** for **only \( A \) and \( B \)**, since **\( W \) is frozen**.

### **Step 1: Compute Loss Gradient**
Define the loss function:
\[
L = \text{LossFunction}(Y_{\text{pred}}, Y_{\text{true}})
\]

Backpropagation starts from:
\[
\frac{\partial L}{\partial Y}
\]

### **Step 2: Compute Gradients for \( A \) and \( B \)**
Using the **chain rule**, we compute:

#### **Gradient w.r.t. \( B \):**
\[
\frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial Y} X
\]
- **Shape of \( A^T \)**: \( (r, D) \)
- **Shape of \( \frac{\partial L}{\partial Y} \)**: \( (B, T, H) \)
- **Shape of \( X \)**: \( (B, T, D) \)
- **Resulting gradient**: \( (r, H) \)

#### **Gradient w.r.t. \( A \):**
\[
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} X B^T
\]
- **Shape of \( B^T \)**: \( (H, r) \)
- **Resulting gradient**: \( (D, r) \)

---

## **4. Weight Updates**
Using **gradient descent**:
\[
A = A - \eta \frac{\partial L}{\partial A}
\]
\[
B = B - \eta \frac{\partial L}{\partial B}
\]
where \( \eta \) is the **learning rate**.

### **What About \( W \)?**
- **\( W \) is frozen**, so it **does not get updated**.
- **Only \( A \) and \( B \) are updated**, making fine-tuning **efficient**.

---

## **5. Computational Complexity Reduction**
### **Standard Fine-Tuning vs. LoRA**
| Approach | Trainable Parameters |
|----------|----------------------|
| Full Fine-Tuning | \( D \times H \) |
| LoRA Fine-Tuning | \( D \times r + r \times H \) |

Since **\( r \ll D, H \)**, LoRA reduces parameters **by orders of magnitude**.

For example:
- **Standard Transformer layer (1266 × 64)** → **80,000 parameters**
- **LoRA with \( r = 8 \)** → **10,000 parameters** (90% reduction!)

---

## **6. PyTorch Implementation**
```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8):
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim, bias=False)  # Frozen weights
        self.A = nn.Linear(input_dim, rank, bias=False)  # Trainable
        self.B = nn.Linear(rank, output_dim, bias=False)  # Trainable

    def forward(self, x):
        return self.W(x) + self.B(self.A(x))  # LoRA: X(W + AB)

# Initialize model
lora_layer = LoRALinear(input_dim=1266, output_dim=64)

# Dummy input
x = torch.randn(4, 128, 1266)  # (Batch, Seq, Features)

# Forward pass
output = lora_layer(x)

# Compute loss
y_true = torch.randn(4, 128, 64)  # Target
loss_fn = nn.MSELoss()
loss = loss_fn(output, y_true)

# Backpropagation
loss.backward()
```

---

## **7. Key Differences Between LoRA and Standard Fine-Tuning**
| Feature | **Standard Fine-Tuning** | **LoRA Fine-Tuning** |
|---------|-----------------|----------------|
| **Memory Usage** | **High** (All weights updated) | **Low** (Only LoRA matrices updated) |
| **Trainable Parameters** | **\( D \times H \)** | **\( D \times r + r \times H \) (Much smaller)** |
| **Computational Cost** | **High** | **Low** |
| **Training Speed** | **Slower** | **Faster** |
| **Model Adaptability** | Full model changes | LoRA adapts pre-trained model |

---

## **8. Summary**
### **How LoRA Works (Step-by-Step)**
1. **Forward Pass**:  
   \[
   Y = X(W + AB)
   \]
2. **Backpropagation**:
   - Compute **gradient w.r.t. \( B \)**:
     \[
     \frac{\partial L}{\partial B} = A^T \frac{\partial L}{\partial Y} X
     \]
   - Compute **gradient w.r.t. \( A \)**:
     \[
     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial Y} X B^T
     \]
3. **Weight Updates**:
   \[
   A = A - \eta \frac{\partial L}{\partial A}, \quad B = B - \eta \frac{\partial L}{\partial B}
   \]
4. **\( W \) is Frozen**: No updates, reducing **memory & computational cost**.

### **Why LoRA is Efficient**
- **Avoids full weight updates**.
- **Uses a low-rank decomposition**.
- **Reduces memory and compute overhead** while maintaining high accuracy.

---

## **9. Conclusion**
- **LoRA fine-tunes models efficiently** by **adding trainable low-rank matrices** instead of modifying full weights.
- **Backpropagation only updates \( A \) and \( B \)**, making training faster and more memory-efficient.
- **LoRA is ideal for large-scale models like BERT and LLaMA**, where full fine-tuning is impractical.

This explanation **mirrors the structure we used for LSTMs and BERT**, ensuring consistency in understanding. Let me know if you need further clarifications!
