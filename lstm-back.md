# **How Backpropagation Works in LSTM (Step-by-Step with Math)**

LSTM (**Long Short-Term Memory**) uses **Backpropagation Through Time (BPTT)** to compute gradients and update weights. Unlike standard RNNs, LSTMs have **gates** that control information flow, making the backpropagation process more complex.

---

## **1. Overview of Backpropagation in LSTM**
The **backpropagation process** in LSTM involves:
1. **Forward Pass:** Compute activations and cell states.
2. **Compute Loss:** Compare output with target.
3. **Backward Pass (BPTT):** Compute gradients for each parameter using the chain rule.
4. **Update Weights:** Apply gradient descent or Adam optimization.

---

## **2. Forward Pass in LSTM**
For each time step \( t \), the LSTM updates its **hidden state** \( h_t \) and **cell state** \( C_t \).

### **Equations for LSTM Gates**
#### **Forget Gate:**
\[
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
\]
Forget gate controls what portion of the past cell state \( C_{t-1} \) is retained.

#### **Input Gate:**
\[
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
\]
Determines how much new information is added.

#### **Candidate Cell State:**
\[
\tilde{C}_t = \tanh(W_C x_t + U_C h_{t-1} + b_C)
\]
Temporary memory value to be added.

#### **Cell State Update:**
\[
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\]
Final cell state after combining past and new information.

#### **Output Gate:**
\[
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
\]
Decides how much of the cell state should be sent to the hidden state.

#### **Hidden State Update:**
\[
h_t = o_t \odot \tanh(C_t)
\]
Final output of the LSTM.

---
## **3. Backpropagation Through Time (BPTT)**
Now, we compute **gradients** for each parameter **by propagating errors backward** from the loss.

### **Step 1: Compute Loss Gradient**
The loss function \( L \) (e.g., **Mean Squared Error**) depends on the final hidden state \( h_T \).

\[
\frac{\partial L}{\partial h_t}
\]
Gradient of the loss with respect to the hidden state.

---

### **Step 2: Compute Gradients for Each Gate**
Using the **chain rule**, we propagate gradients back **through gates**.

#### **Gradient w.r.t Output Gate \( o_t \):**
\[
\frac{\partial L}{\partial o_t} = \frac{\partial L}{\partial h_t} \odot \tanh(C_t)
\]
\[
\frac{\partial L}{\partial W_o} = \frac{\partial L}{\partial o_t} \cdot x_t^T
\]
\[
\frac{\partial L}{\partial U_o} = \frac{\partial L}{\partial o_t} \cdot h_{t-1}^T
\]

#### **Gradient w.r.t Cell State \( C_t \):**
\[
\frac{\partial L}{\partial C_t} = \frac{\partial L}{\partial h_t} \odot o_t \odot (1 - \tanh^2(C_t))
\]
\[
\frac{\partial L}{\partial C_t} += f_{t+1} \cdot \frac{\partial L}{\partial C_{t+1}}  \quad \text{(gradient from next time step)}
\]

#### **Gradient w.r.t Forget Gate \( f_t \):**
\[
\frac{\partial L}{\partial f_t} = \frac{\partial L}{\partial C_t} \odot C_{t-1}
\]
\[
\frac{\partial L}{\partial W_f} = \frac{\partial L}{\partial f_t} \cdot x_t^T
\]
\[
\frac{\partial L}{\partial U_f} = \frac{\partial L}{\partial f_t} \cdot h_{t-1}^T
\]

#### **Gradient w.r.t Input Gate \( i_t \):**
\[
\frac{\partial L}{\partial i_t} = \frac{\partial L}{\partial C_t} \odot \tilde{C}_t
\]
\[
\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial i_t} \cdot x_t^T
\]
\[
\frac{\partial L}{\partial U_i} = \frac{\partial L}{\partial i_t} \cdot h_{t-1}^T
\]

#### **Gradient w.r.t Candidate Cell State \( \tilde{C}_t \):**
\[
\frac{\partial L}{\partial \tilde{C}_t} = \frac{\partial L}{\partial C_t} \odot i_t \odot (1 - \tilde{C}_t^2)
\]
\[
\frac{\partial L}{\partial W_C} = \frac{\partial L}{\partial \tilde{C}_t} \cdot x_t^T
\]
\[
\frac{\partial L}{\partial U_C} = \frac{\partial L}{\partial \tilde{C}_t} \cdot h_{t-1}^T
\]

---

## **4. Weight Updates**
Once gradients are computed, weights are updated using **gradient descent**:

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

where \( \eta \) is the **learning rate**.

---

## **5. Computational Complexity of BPTT in LSTM**
- **Each time step requires \( O(H^2) \) operations**.
- **Total complexity per sequence:** \( O(T \cdot H^2) \).
- **Compared to Transformers (BERT)**, LSTMs are **slower** because they **backpropagate through time** instead of **through layers**.

---

## **6. Key Differences Between LSTM and Transformer Backpropagation**
| Feature | **LSTM (BPTT)** | **BERT (Backprop Through Layers)** |
|---------|----------------|----------------|
| **Gradient Flow** | **Backpropagates through time** | **Backpropagates through layers** |
| **Long-Term Dependencies** | Harder to train (**gradient vanishing**) | No issue (**self-attention spans full sequence**) |
| **Computational Complexity** | \( O(T \cdot H^2) \) (Sequential updates) | \( O(T^2 \cdot H) \) (Parallel updates) |
| **Memory Usage** | Lower (only stores past states) | Higher (stores full attention scores) |
| **Training Speed** | **Slow** (Sequential BPTT) | **Fast** (Parallel optimization) |

---

## **7. PyTorch Code for LSTM Backpropagation**
```python
import torch
import torch.nn as nn

# Define LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last time step output
        return out

# Initialize model, loss function, optimizer
model = SimpleLSTM(input_size=1266, hidden_size=64, num_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy input and target
x = torch.randn(4, 128, 1266)  # (Batch Size, Sequence Length, Input Dim)
y_true = torch.randn(4, 1)

# Forward Pass
y_pred = model(x)
loss = criterion(y_pred, y_true)

# Backpropagation
loss.backward()
optimizer.step()
```

---

## **8. Conclusion**
- **LSTM backpropagation uses BPTT**, meaning gradients flow **through multiple time steps**.
- **Gate derivatives are computed using the chain rule**.
- **Gradient vanishing/exploding can happen**, requiring **gradient clipping**.
- **Compared to Transformers (BERT)**, LSTMs are **slower and harder to train** for long sequences.

This explains **LSTM backpropagation mathematically and computationally**, just like we did for BERT. Let me know if you need more clarifications!
