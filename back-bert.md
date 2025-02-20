### **How Backpropagation Works in LSTM vs. BERT**
Backpropagation in both **LSTM** and **BERT** follows the **chain rule of differentiation**, also known as **backpropagation through time (BPTT)** for LSTMs and **backpropagation through layers** for BERT. The core steps include:

1. **Forward Pass** – Compute activations and outputs.
2. **Compute Loss** – Compare the output with the actual target.
3. **Backward Pass (Backpropagation)** – Compute gradients using the chain rule.
4. **Update Weights** – Use an optimizer (e.g., SGD, Adam) to adjust parameters.

Now, let's dive into **each step mathematically** for **LSTM and BERT**.

---

## **1. Backpropagation in LSTM**
LSTM backpropagation works **through time** (BPTT), meaning gradients **flow backward** across multiple time steps.

### **Step 1: Forward Pass**
At each time step \( t \), the LSTM computes:

1. **Forget Gate**:
   \[
   f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
   \]
2. **Input Gate**:
   \[
   i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
   \]
3. **Candidate Cell State**:
   \[
   \tilde{C}_t = \tanh(W_C x_t + U_C h_{t-1} + b_C)
   \]
4. **Cell State Update**:
   \[
   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
   \]
5. **Output Gate**:
   \[
   o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
   \]
6. **Hidden State Update**:
   \[
   h_t = o_t \odot \tanh(C_t)
   \]

### **Step 2: Compute Loss**
The loss \( L \) is computed as:

\[
L = \text{LossFunction}(y_{\text{pred}}, y_{\text{true}})
\]

---

### **Step 3: Backpropagation Through Time (BPTT)**
BPTT **unfolds the LSTM across time** and computes **gradients** backward from the loss.

#### **Gradient of the Loss w.r.t. Hidden State \( h_t \)**
\[
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_{t+1}} \cdot \frac{\partial h_{t+1}}{\partial h_t}
\]

Using the **chain rule**, we backpropagate gradients through:
1. **Output Gate**:
   \[
   \frac{\partial L}{\partial o_t} = \frac{\partial L}{\partial h_t} \odot \tanh(C_t)
   \]
2. **Cell State**:
   \[
   \frac{\partial L}{\partial C_t} = \frac{\partial L}{\partial h_t} \odot o_t \odot (1 - \tanh^2(C_t))
   \]
   Plus the **gradient from the next time step**:
   \[
   \frac{\partial L}{\partial C_t} += f_{t+1} \cdot \frac{\partial L}{\partial C_{t+1}}
   \]
3. **Forget, Input, and Candidate Cell Gates**:
   \[
   \frac{\partial L}{\partial f_t} = \frac{\partial L}{\partial C_t} \odot C_{t-1}
   \]
   \[
   \frac{\partial L}{\partial i_t} = \frac{\partial L}{\partial C_t} \odot \tilde{C}_t
   \]
   \[
   \frac{\partial L}{\partial \tilde{C}_t} = \frac{\partial L}{\partial C_t} \odot i_t \odot (1 - \tilde{C}_t^2)
   \]

#### **Weight Updates**
Using **gradient descent**:
\[
W_f = W_f - \eta \frac{\partial L}{\partial W_f}
\]
\[
W_i = W_i - \eta \frac{\partial L}{\partial W_i}
\]
\[
W_C = W_C - \eta \frac{\partial L}{\partial W_C}
\]
\[
W_o = W_o - \eta \frac{\partial L}{\partial W_o}
\]

where \( \eta \) is the learning rate.

---

## **2. Backpropagation in BERT**
Unlike LSTMs, BERT **does not use recurrence**, meaning backpropagation happens **layer-wise** rather than through time.

### **Step 1: Forward Pass**
BERT computes:
1. **Token Embeddings \( X \)**
2. **Self-Attention Scores**:
   \[
   A = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
   \]
3. **Feedforward Network**:
   \[
   \text{FFN}(X) = \max(0, XW_1 + b_1) W_2 + b_2
   \]

### **Step 2: Compute Loss**
Same as LSTM:
\[
L = \text{LossFunction}(y_{\text{pred}}, y_{\text{true}})
\]

---

### **Step 3: Backpropagation Through Layers**
Since **each BERT layer is independent**, gradients **flow backward layer by layer** instead of through time.

#### **Gradient of the Loss w.r.t. the Output Layer**
\[
\frac{\partial L}{\partial H^{(L)}}
\]
Backpropagating through each **Transformer block**:
1. **Through the Feedforward Network**:
   \[
   \frac{\partial L}{\partial W_2} = H^{(L-1)} \cdot \frac{\partial L}{\partial H^{(L)}}
   \]
   \[
   \frac{\partial L}{\partial W_1} = X^{(L-1)} \cdot \frac{\partial L}{\partial H^{(L-1)}}
   \]
2. **Through Self-Attention**:
   \[
   \frac{\partial L}{\partial Q} = \frac{\partial L}{\partial A} \cdot K
   \]
   \[
   \frac{\partial L}{\partial K} = Q \cdot \frac{\partial L}{\partial A}
   \]
   \[
   \frac{\partial L}{\partial V} = \frac{\partial L}{\partial A} \cdot A
   \]

#### **Weight Updates**
Using **gradient descent**:
\[
W_Q = W_Q - \eta \frac{\partial L}{\partial W_Q}
\]
\[
W_K = W_K - \eta \frac{\partial L}{\partial W_K}
\]
\[
W_V = W_V - \eta \frac{\partial L}{\partial W_V}
\]

---

## **3. Key Differences Between LSTM and BERT Backpropagation**
| Feature | **LSTM (BPTT)** | **BERT (Backprop Through Layers)** |
|---------|----------------|----------------|
| **Gradient Flow** | Backpropagates **through time** (many steps) | Backpropagates **through layers** |
| **Long-Term Dependencies** | Harder to train (gradient vanishing) | No issue (Self-attention spans entire sequence) |
| **Computational Complexity** | **O(T × H²)** (sequential updates) | **O(T² × H)** (parallel updates) |
| **Memory Usage** | Low (only past states stored) | High (Stores full attention scores) |
| **Training Speed** | **Slow** (sequential updates) | **Fast** (parallel updates) |

---

## **4. Conclusion**
- **LSTM uses Backpropagation Through Time (BPTT)**, meaning gradients flow **through past time steps**.
- **BERT backpropagates through layers**, allowing **parallel updates**.
- **BERT is more efficient** than LSTM because **self-attention allows faster gradient computation**.

This explains **how backpropagation works in LSTM and BERT step by step**, just like we did for the forward pass. Let me know if you need further clarifications!
