# Sequence Models and Transformers

Welcome to the module on sequence models and transformers. This folder covers the architectures and training methods that power every modern AI system — from the RNNs that introduced the concept of memory, to the transformers that dominate language, vision, and beyond.

---

### Lesson 1: Sequence Models — Giving Machines Memory

*Real-world example:* A translation model converting a 100-word English sentence into French must remember the subject introduced in the first sentence to correctly translate verbs in the last. Feedforward networks cannot do this — they have no notion of order or memory.

#### Why Sequences Are Different

In a standard feedforward network, the input is a fixed-size vector, and the output is a fixed-size vector. Order does not matter: a neural network that sees `[1, 2, 3]` produces the same result as `[3, 1, 2]`.

##### The Math Behind Order-Invariance

A feedforward layer computes $y = f(W \cdot x + b)$ for every input $x$. There is no term in this expression that depends on *which position* $x_i$ came from — only on the value at that position. Concretely, for two inputs that are permutations of each other:

$$
\begin{aligned}
\mathbf{x}_A &= [1,\ 2,\ 3] \\
\mathbf{x}_B &= [3,\ 1,\ 2] \quad \text{(a permutation of A)} \\
W &= \begin{bmatrix} 0.5 & 0.0 & 0.0 \\ 0.0 & 0.5 & 0.0 \\ 0.0 & 0.0 & 0.5 \end{bmatrix}, \quad b = [0] \\
\mathbf{y}_A &= W\,\mathbf{x}_A = [0.5,\ 1.0,\ 1.5] \\
\mathbf{y}_B &= W\,\mathbf{x}_B = [1.5,\ 0.5,\ 1.0] \quad \text{(DIFFERENT values, but both are just weighted sums)}
\end{aligned}
$$

*The model has no signal to know that A and B are "the same sequence in a different order."*

The same value set produces different outputs because each input position is tied to a specific weight — but the network has *no mechanism* to detect that the two sequences carry the same set of values in swapped order. To a feedforward net, `[1, 2, 3]` and `[3, 1, 2]` are just two unrelated vectors that happen to share the same bag of values.

- **The core property:** A function $f$ is *permutation-invariant* if $f(\mathrm{permute}(x)) = f(x)$ for every permutation, and *permutation-equivariant* if $f(\mathrm{permute}(x)) = \mathrm{permute}(f(x))$. A plain feedforward layer is permutation-invariant over its input (the output is the same regardless of order). Bag-of-Words, sum-pooling, and mean-pooling are also permutation-invariant — which is why they cannot distinguish "dog bites man" from "man bites dog."

##### Concrete Example: The "Dog Bites Man" Problem

Run two sentences through a Bag-of-Words vectorizer and the results are *identical* — a feedforward classifier downstream sees the exact same features:

```
Sentence A: "the dog bites the man"   →  BoW = {the: 2, dog: 1, bites: 1, man: 1}
Sentence B: "the man bites the dog"   →  BoW = {the: 2, dog: 1, bites: 1, man: 1}

Feedforward output: P(A) = P(B) = 0.50       ← The model literally cannot tell them apart
```

This is the same root cause as the $[1, 2, 3]$ example: a function that does not look at *position* treats any reordering of the same set as the same input.

*Real-world example:* A sentiment classifier built on top of Bag-of-Words features rates *"this movie was not good"* and *"this good movie was not"* as **identical** — even though a human reads the first as mildly negative and the second as confusing or self-contradictory. The classifier has no way to know that "not" appearing *before* "good" is what flips the sentiment.

But most of the data we care about is *sequential*:

- **Text:** A sequence of tokens. "Dog bites man" ≠ "Man bites dog."
- **Audio:** A sequence of amplitude samples.
- **Video:** A sequence of frames.
- **Time series:** Stock prices, sensor readings, server logs.
- **Code:** A sequence of tokens that must compile.

For a model to understand these, it must encode *position* — not just *content*.

#### Markov Chains — The Simplest Memory Model

The simplest model of sequence memory is a **Markov chain**: the next state depends only on the current state.

$$
P(X_t \mid X_{t-1},\ X_{t-2},\ \ldots) = P(X_t \mid X_{t-1})
$$

This works for short, structured sequences (weather, simple games) but fails for language, where the meaning of a word can depend on words 50 tokens earlier.

A model that handles sequences well must: process inputs one at a time, maintain a *summary* of what it has seen so far, update that summary as new inputs arrive, and use the summary to make predictions. This is exactly what a Recurrent Neural Network (RNN) does.

#### Recurrent Neural Networks (RNNs)

The architecture of a vanilla RNN, shown below, makes the three pieces of the formulation explicit: the input $x_t$, the previous hidden state $h_{t-1}$, the new hidden state $h_t$, and the output $y_t$ — all wired together with the shared weight matrices $W_{xh}$, $W_{hh}$, and $W_{hy}$.

![RNN architecture diagram: at each time step, the input $x_t$ and the previous hidden state $h_{t-1}$ are combined through the shared weights $W_{xh}$ and $W_{hh}$ to produce the new hidden state $h_t$, which is then projected through $W_{hy}$ to produce the output $y_t$. The same matrices are reused at every time step.](RNN%20Architecture.png)

**Reading the diagram:** each time-step box in the picture applies the *same* transformation:
1. The current input $x_t$ enters from the bottom.
2. The previous hidden state $h_{t-1}$ enters from the left (it is the "memory" carried over from the previous step).
3. The two are combined through the shared weights $W_{xh}$ and $W_{hh}$ to produce the new hidden state $h_t$.
4. $h_t$ is projected through the shared output weight $W_{hy}$ to produce $y_t$.
5. $h_t$ is also passed to the right, becoming $h_{t-1}$ for the next time step.

The boxes at every time step are **identical** — the only things that change are the inputs $x_t$ and the incoming hidden state $h_{t-1}$. This is the visual counterpart of the weight-sharing principle we will state formally in the next section.

#### Intuition Behind RNN

An RNN processes a sequence **one element at a time**, maintaining a **hidden state** (the "memory") that is updated after every input. The hidden state carries information from previous time steps forward in time.

*Example:* Reading the sentence *"I love machine learning"* one word at a time:

$$
\begin{array}{rcl}
\text{Word 1: } x_0 = \text{``I''} & \longrightarrow & h_0 = \tanh\!\left(W_{hh}\, h_{-1} + W_{xh}\, x_0 + b_h\right) \\
\text{Word 2: } x_1 = \text{``love''} & \longrightarrow & h_1 = \tanh\!\left(W_{hh}\, h_0 + W_{xh}\, x_1 + b_h\right) \\
\text{Word 3: } x_2 = \text{``machine''} & \longrightarrow & h_2 = \tanh\!\left(W_{hh}\, h_1 + W_{xh}\, x_2 + b_h\right) \\
\text{Word 4: } x_3 = \text{``learning''} & \longrightarrow & h_3 = \tanh\!\left(W_{hh}\, h_2 + W_{xh}\, x_3 + b_h\right)
\end{array}
$$

At each step the same operation is applied — only the input and previous hidden state change. The recurrent weight $W_{hh}$ is the *same matrix* in every equation; it is the "memory filter" that decides what to keep and what to forget. The input weight $W_{xh}$ is also identical at every step.

| Step | Word | Updated Memory | What $h_t$ encodes |
|---|---|---|---|
| 1 | `I` | $h_0$ | the first word |
| 2 | `love` | $h_1$ | "I love" |
| 3 | `machine` | $h_2$ | "I love machine" |
| 4 | `learning` | $h_3$ | "I love machine learning" |

After processing the full sentence, $h_3$ is a single fixed-size vector that **encodes the meaning of the entire sentence so far** — exactly what a downstream classifier, sentiment scorer, or decoder needs.

> *Think of the hidden state as a rolling summary.* Each new word refines the summary, but the summary never grows in size. It is always a vector of the same dimension (e.g., 128 or 512 numbers), no matter how long the sentence is. This is how an RNN can handle variable-length input with a fixed-size representation.

*Real-world example:* When you read *"Despite the rain, she went outside without an umbrella, confident the clouds would pass"*, by the time you reach the period your mental "summary" carries the subject (she), the contrast (despite the rain), the action (went outside), and the resolution (clouds would pass). An RNN's hidden state is the same idea — a single vector that *accumulates* meaning as it reads.

#### Mathematical Formulation

The RNN's behavior is fully specified by two equations. At every time step, the **current hidden state** depends on exactly two things:

- The **current input** $x_t$
- The **previous hidden state** $h_{t-1}$

**Core hidden-state equation** — the memory update:

$$
h_t = \tanh\!\left(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h\right)
$$

| Symbol | Meaning |
|---|---|
| $h_t$ | current hidden state — the new "memory" after reading $x_t$ |
| $h_{t-1}$ | previous hidden state — the "memory" carried in from the last time step |
| $x_t$ | current input (e.g., the embedding of the $t$-th word) |
| $W_{hh}$ | hidden-to-hidden weights — the "memory filter" applied to $h_{t-1}$ |
| $W_{xh}$ | input-to-hidden weights — the "input filter" applied to $x_t$ |
| $b_h$ | bias term |

**Output equation** — generate the output from the current memory:

$$
y_t = W_{hy} \cdot h_t + b_y
$$

| Symbol | Meaning |
|---|---|
| $y_t$ | output at time $t$ (e.g., class logits, next-token scores) |
| $W_{hy}$ | hidden-to-output weights — projects the hidden state $h_t$ into output space |
| $b_y$ | output bias |

**The complete RNN in two lines:**

$$
\begin{aligned}
h_t &= \tanh\!\left(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h\right) && \text{(memory update)} \\
y_t &= W_{hy} \cdot h_t + b_y && \text{(output from memory)}
\end{aligned}
$$

These two equations — one for the **memory update** and one for the **output** — are the complete specification of a vanilla RNN. Everything else in this chapter (LSTM, GRU, attention, transformers) is a more sophisticated version of this same idea.

#### Why Same Weights Are Used?

In a traditional (non-recurrent) network, applying a different transformation at each position would mean learning a *separate* set of weights for every time step. An RNN takes the opposite approach: **the same weight matrices are shared across every time step**.

**What a traditional (non-recurrent) approach would look like** — different weights per step:

$$
\begin{array}{rcl}
\text{Step 1: } & h_1 = \tanh\!\left(W^{(1)}_{hh}\, h_0 + W^{(1)}_{xh}\, x_1 + b^{(1)}_h\right) \\
\text{Step 2: } & h_2 = \tanh\!\left(W^{(2)}_{hh}\, h_1 + W^{(2)}_{xh}\, x_2 + b^{(2)}_h\right) \\
\text{Step 3: } & h_3 = \tanh\!\left(W^{(3)}_{hh}\, h_2 + W^{(3)}_{xh}\, x_3 + b^{(3)}_h\right) \\
\vdots & & \\
\text{Step } T: & h_T = \tanh\!\left(W^{(T)}_{hh}\, h_{T-1} + W^{(T)}_{xh}\, x_T + b^{(T)}_h\right)
\end{array}
$$

**What an RNN actually does** — the *same* weights at every step:

$$
\begin{array}{rcl}
\text{Step 1: } & h_1 = \tanh\!\left(W_{hh}\, h_0 + W_{xh}\, x_1 + b_h\right) \\
\text{Step 2: } & h_2 = \tanh\!\left(W_{hh}\, h_1 + W_{xh}\, x_2 + b_h\right) \\
\text{Step 3: } & h_3 = \tanh\!\left(W_{hh}\, h_2 + W_{xh}\, x_3 + b_h\right) \\
\vdots & & \\
\text{Step } T: & h_T = \tanh\!\left(W_{hh}\, h_{T-1} + W_{xh}\, x_T + b_h\right)
\end{array}
$$

The superscripts are gone — $W_{hh}$, $W_{xh}$, $W_{hy}$ and $b_h$ are the **same matrices and vector at every step**. This weight-sharing is what makes an RNN *recurrent*.

**Why this matters — three reasons:**

1. **Temporal consistency.** The model treats time steps symmetrically. The transformation that turns $h_{t-1}$ and $x_t$ into $h_t$ is the same at every position, so the same word always updates the memory in the same way regardless of where it appears in the sequence.

2. **Variable-length sequences.** Because the same weights apply at every step, an RNN trained on 50-word sentences can process a 5-word or a 500-word sequence with no architectural change. The number of parameters is *independent* of the sequence length.

3. **Drastically fewer parameters.** A naive position-specific approach needs $T$ separate copies of every weight matrix. For a 100-step sequence with $d_h = 128$, that is $100 \times (128 \times 128) = 1{,}638{,}400$ parameters for $W_{hh}$ alone — versus $128 \times 128 = 16{,}384$ for the shared version. **The shared-weight design cuts parameter count by a factor of $T$.**

*Real-world example:* The shared-weight design is what makes RNNs (and their descendants, LSTMs and transformers) practical at all. Without it, a model that processes a paragraph would have a thousand times more parameters than one that processes a sentence — and would need a thousand times more data to train.

#### Unrolling Through Time

Visually, an RNN is a loop, but computationally we "unroll" it through time:

$$
\begin{array}{ccccc}
x_0 & \xrightarrow{\,W_{xh}\,} & h_0 & \xrightarrow{\,W_{hy}\,} & y_0 \\
    &                         & \big\downarrow\; W_{hh}       &        &     \\
x_1 & \xrightarrow{\,W_{xh}\,} & h_1 & \xrightarrow{\,W_{hy}\,} & y_1 \\
    &                         & \big\downarrow\; W_{hh}       &        &     \\
x_2 & \xrightarrow{\,W_{xh}\,} & h_2 & \xrightarrow{\,W_{hy}\,} & y_2 \\
    &                         & \big\downarrow\; W_{hh}       &        &     \\
x_3 & \xrightarrow{\,W_{xh}\,} & h_3 & \xrightarrow{\,W_{hy}\,} & y_3
\end{array}
$$

Reading the diagram: at each time step $t$, the input $x_t$ is transformed into $h_t$ by $W_{xh}$, the hidden state $h_t$ is transformed into the output $y_t$ by $W_{hy}$, and $h_t$ is also passed to the next time step (via $W_{hh}$) to become $h_{t+1}$'s input. Each column is the *same* network with the *same* weights, just applied at a different time step. The hidden state $h_t$ flows forward in time, carrying information forward.

#### The Vanishing Gradient Problem

RNNs are trained with **Backpropagation Through Time (BPTT)** — gradients flow backward from the last time step to the first.

The gradient at time step $t$ is roughly:

$$
\frac{\partial L}{\partial h_t} = \frac{\partial L}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
$$

That product of Jacobians is the problem. If the largest eigenvalue of $\frac{\partial h_k}{\partial h_{k-1}}$ is less than 1, the product **shrinks exponentially**:

$$
0.9 \times 0.9 \times 0.9 \times \cdots \times 0.9 \;\;(\text{100 times}) = 0.9^{100} \approx 0.000026
$$

$$
0.5 \times 0.5 \times 0.5 \times \cdots \times 0.5 \;\;(\text{100 times}) = 0.5^{100} \approx 0
$$

After 100 time steps, the gradient signal from the end of the sequence has effectively **vanished** when it reaches the beginning. The network literally cannot learn long-range dependencies because the training signal does not reach the early weights.

This is the **vanishing gradient problem**, and it is the central failure mode of vanilla RNNs.

#### Understanding Vanishing and Exploding Gradients in RNN

The vanishing-gradient result above is worth unpacking in detail, because the *exploding* version of the same problem is just as destructive. Both come from the same multiplicative structure; they only differ in whether the repeated multiplier is smaller or greater than 1.

##### First, Understand What a Gradient Is

During training, the model makes predictions. The prediction has some **error (loss)**. The **gradient** tells the model two things:

- **Which weights** caused the error.
- **How much** each weight should be updated.

*Think of the gradient as a learning signal.* Its magnitude directly controls how fast the network learns:

| Gradient magnitude | Effect on learning |
|---|---|
| **Large gradient** | Learn a lot from this example — adjust weights significantly |
| **Small gradient** | Learn a little — adjust weights only slightly |
| **Zero gradient** | Learn nothing — weights do not change at all |

##### Why Does This Problem Occur in RNN?

Consider a sequence of inputs processed one at a time:

$$
x_1 \;\to\; x_2 \;\to\; x_3 \;\to\; x_4 \;\to\; x_5
$$

*Example sentence:* *"The movie released in 1990 was a blockbuster."*

During training, the prediction error is computed at the last word (the final output $y_T$). To update the weights that produced $h_1$ (the hidden state after reading *"The"*), that error must flow **backward** through every intermediate time step:

$$
\underbrace{x_1}_{\text{word 1}} \;\xleftarrow{\;\nabla\;}\; \underbrace{x_2}_{\text{word 2}} \;\xleftarrow{\;\nabla\;}\; \underbrace{x_3}_{\text{word 3}} \;\xleftarrow{\;\nabla\;}\; \underbrace{x_4}_{\text{word 4}} \;\xleftarrow{\;\nabla\;}\; \underbrace{x_5}_{\text{word 5}} \;\xleftarrow{\;\nabla\;}\; \text{Loss}
$$

This process is called **Backpropagation Through Time (BPTT)**. The gradient at step $t$ depends on the product of Jacobians from every later step — which is exactly the product that explodes or vanishes.

##### The Vanishing Gradient Problem

###### What Happens?

While moving backward, the gradient is **repeatedly multiplied by the same weight value** (the recurrent weight $W_{hh}$). Suppose that weight value is:

$$
w = 0.5
$$

Then the gradient shrinks by a factor of 0.5 at every step:

| Step | Gradient | Calculation |
|---|---|---|
| 1 | $1.0$ | (start) |
| 2 | $0.5$ | $1.0 \times 0.5$ |
| 3 | $0.25$ | $0.5 \times 0.5$ |
| 4 | $0.125$ | $0.25 \times 0.5$ |
| 5 | $0.0625$ | $0.125 \times 0.5$ |
| 6 | $0.03125$ | $0.0625 \times 0.5$ |

As we move further back, the gradient compounds multiplicatively:

$$
\begin{aligned}
0.5^{10} &\approx 0.000977 \\
0.5^{20} &\approx 0.00000095 \\
0.5^{100} &\approx 7.9 \times 10^{-31} \quad (\text{effectively zero})
\end{aligned}
$$

**The earlier words receive almost no learning signal.**

###### Visualization

$$
\begin{array}{rcl}
\text{Prediction Error} & \longrightarrow & \text{Gradient} = 1.0 \\
& \searrow & \\
x_5 & \xleftarrow{\;\nabla = 1.0\;} & \\
x_4 & \xleftarrow{\;\nabla = 0.5\;} & \\
x_3 & \xleftarrow{\;\nabla = 0.25\;} & \\
x_2 & \xleftarrow{\;\nabla = 0.125\;} & \\
x_1 & \xleftarrow{\;\nabla = 0.0625\;} & \\
\end{array}
$$

The gradient value at the leftmost position is tiny — the weights that processed $x_1$ get an almost-zero update, so they barely learn anything.

###### What Does the Model Experience?

The model effectively says:

> *"I can't see the information from the beginning anymore."*

*Example:*

> *The movie released in 1990 and shown in many countries became extremely popular and it was a blockbuster.*

To understand the word **"blockbuster"**, the model may need information from the phrase **"released in 1990"** — but that information occurred many time steps earlier. Due to vanishing gradients, the gradient reaching the words *"released in 1990"* is approximately zero. The model **cannot learn that relationship** no matter how much data it sees.

###### Consequences of Vanishing Gradients

- **Earlier information is forgotten** — the model effectively has a short memory.
- **Long-term dependencies cannot be learned** — relationships that span more than ~10 time steps are invisible to training.
- **Training becomes very slow** — early weights barely update, even after many epochs.
- **The model behaves as if it only remembers recent inputs** — its effective context window collapses to a few tokens.

##### The Exploding Gradient Problem

The mirror-image failure mode happens when the recurrent weight $|w| > 1$. The gradient then **grows** by a factor of $w$ at every step instead of shrinking.

###### What Happens?

Suppose the weight value is:

$$
w = 2
$$

Then the gradient **doubles** at every step:

| Step | Gradient | Calculation |
|---|---|---|
| 1 | $1.0$ | (start) |
| 2 | $2$ | $1.0 \times 2$ |
| 3 | $4$ | $2 \times 2$ |
| 4 | $8$ | $4 \times 2$ |
| 5 | $16$ | $8 \times 2$ |
| 6 | $32$ | $16 \times 2$ |

As we go further back, the gradient compounds multiplicatively the other direction:

$$
\begin{aligned}
2^{10} &= 1{,}024 \\
2^{20} &= 1{,}048{,}576 \\
2^{100} &\approx 1.27 \times 10^{30} \quad (\text{astronomically large})
\end{aligned}
$$

**The gradient becomes enormous.**

###### Visualization

$$
\begin{array}{rcl}
\text{Prediction Error} & \longrightarrow & \text{Gradient} = 1.0 \\
& \searrow & \\
x_5 & \xleftarrow{\;\nabla = 1.0\;} & \\
x_4 & \xleftarrow{\;\nabla = 2\;} & \\
x_3 & \xleftarrow{\;\nabla = 4\;} & \\
x_2 & \xleftarrow{\;\nabla = 8\;} & \\
x_1 & \xleftarrow{\;\nabla = 16\;} & \\
\end{array}
$$

Instead of becoming smaller as it travels backward, the gradient **grows** at every step.

###### What Happens During Weight Updates?

The standard gradient-descent update rule is:

$$
W_{\text{new}} = W_{\text{old}} - \eta \cdot \nabla L
$$

where $\eta$ is the learning rate. Suppose:

$$
W_{\text{old}} = 5, \quad \eta = 0.01, \quad \nabla L = 1{,}000{,}000
$$

The update is:

$$
\begin{aligned}
W_{\text{new}} &= 5 - (0.01 \times 1{,}000{,}000) \\
              &= 5 - 10{,}000 \\
              &= -9{,}995
\end{aligned}
$$

A **massive jump** in the wrong direction.

###### Consequences of Exploding Gradients

- **Weights become extremely large** (or extremely negative) in a single update.
- **Training becomes unstable** — the loss curve oscillates wildly between epochs.
- **The model may never converge** — every update overshoots a reasonable solution.
- **Sometimes the loss becomes** $\infty$ **or** $\mathrm{NaN}$ — at which point training **fails completely**.

##### Vanishing vs. Exploding — A Side-by-Side Comparison

| | Vanishing ($|w| < 1$) | Exploding ($|w| > 1$) |
|---|---|---|
| Multiplier example | $0.5$ | $2$ |
| Gradient at step 10 | $0.5^{10} \approx 10^{-3}$ | $2^{10} = 1{,}024$ |
| Gradient at step 100 | $0.5^{100} \approx 0$ | $2^{100} \approx 10^{30}$ |
| Effect on early weights | Almost no update | Wild, unstable updates |
| Training behavior | Slow, stalls | Diverges, may NaN |
| Mitigations | LSTM/GRU gating, residual paths, better init | **Gradient clipping** (cap $\nabla L$ to a max norm) |

> *Both problems are solved by the LSTM.* The next section introduces its gating mechanism, which replaces the single multiplicative recurrence with an **additive** cell-state update — and that one design choice is what restores stable gradient flow across hundreds of time steps.

#### LSTM — Long Short-Term Memory

The LSTM, introduced by Hochreiter and Schmidhuber in 1997, solves the vanishing gradient problem with one key insight: **add a separate "memory highway" that information can flow through with minimal transformation.**

The LSTM maintains two vectors at each time step:
- `h_t` = hidden state (short-term, used for output)
- `C_t` = cell state (long-term memory, the "highway")

Three **gates** control the flow:

**Forget gate** — what to throw away from the cell state:
$$
f_t = \sigma\!\left(W_f \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_f\right)
$$

**Input gate** — what new information to store:
$$
\begin{aligned}
i_t &= \sigma\!\left(W_i \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_i\right) \\
\tilde{C}_t &= \tanh\!\left(W_C \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_C\right)
\end{aligned}
$$

**Cell state update** — the highway:
$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

**Output gate** — what to read out from the cell state:
$$
\begin{aligned}
o_t &= \sigma\!\left(W_o \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_o\right) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

The $\odot$ symbol is element-wise multiplication. The $\sigma$ is the sigmoid function, which squashes values to $[0, 1]$ — a value of 0 means "let nothing through," 1 means "let everything through."

The cell state update is the magic: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$ is an **additive** update, not a multiplicative one. Gradients flow through addition, which preserves them across long time spans. The forget gate can choose to "let the gradient through" by setting $f_t \approx 1$.

#### LSTM Gate Flow

$$
\begin{array}{rcl}
x_t,\; h_{t-1} & \xrightarrow{\text{concat}} & 
\underbrace{\begin{array}{c} f_t \;=\; \sigma(\cdot) \\ i_t \;=\; \sigma(\cdot) \\ \tilde{C}_t \;=\; \tanh(\cdot) \end{array}}_{\text{three gates from }[x_t, h_{t-1}]} \\[1.2em]
& & \downarrow\;\times \\[0.3em]
C_{t-1} & \xrightarrow{\;\;\times\;f_t\;\;} & 
\underbrace{C_t \;=\; f_t \odot C_{t-1} \;+\; i_t \odot \tilde{C}_t}_{\text{cell state update}} 
\;\xrightarrow{\;\tanh\;}\; 
\underbrace{h_t \;=\; o_t \odot \tanh(C_t)}_{\text{hidden state output}} \\[0.6em]
& & o_t \;\xleftarrow{\;\;\sigma(\cdot)\;\;}\; [x_t, h_{t-1}]
\end{array}
$$

#### Worked Example: Pronoun Resolution

Suppose an LSTM is reading:  
*"The PM of India gave a speech. **He** spoke about the economy."*

After reading "PM of India," the cell state should encode the subject. When the model encounters "He" 30 words later, it needs to:
1. **Forget** the now-irrelevant recent context
2. **Read** the cell state to recall "PM of India"
3. **Output** "He" (the pronoun)

A vanilla RNN cannot do this — the gradient from the loss on the word "He" cannot reach back to update the weights that processed "PM of India" 30 steps earlier. An LSTM can, because the cell state preserves that information through an additive path.

*Real-world example:* Google's Smart Reply (2016) used LSTMs to generate short email responses by tracking conversational context across multiple sentences.

#### GRU — The Simplified Alternative

The **Gated Recurrent Unit** (Cho et al., 2014) simplifies the LSTM to two gates:

$$
\begin{aligned}
z_t &= \sigma\!\left(W_z \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_z\right) && \text{(update gate)} \\
r_t &= \sigma\!\left(W_r \cdot \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix} + b_r\right) && \text{(reset gate)} \\
\tilde{h}_t &= \tanh\!\left(W \cdot \begin{bmatrix} r_t \odot h_{t-1} \\ x_t \end{bmatrix} + b\right) && \text{(candidate)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(interpolation)}
\end{aligned}
$$

The GRU combines the forget and input gates into a single **update gate** $z_t$ and merges the cell state and hidden state. It has fewer parameters, trains faster, and often performs comparably to LSTM on many tasks.

#### RNN vs LSTM vs GRU

| Feature | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | 0 | 3 (forget, input, output) | 2 (update, reset) |
| State vectors | 1 (h) | 2 (h, C) | 1 (h) |
| Parameters | Fewest | Most | Middle |
| Long-range memory | Poor | Excellent | Very good |
| Training speed | Fastest | Slowest | Middle |
| When to use | Short sequences | Long sequences, strong baseline | Good default choice |

#### Seq2Seq — Encoder-Decoder Pattern

For tasks like machine translation, the input and output have *different lengths*. "How are you?" (3 words) → "Comment ça va?" (3 words) — close. But "I love this book" → "J'adore ce livre" — different word order, different words.

The **Seq2Seq** architecture (Sutskever et al., 2014) handles this with two RNNs:

1. **Encoder:** Reads the input sequence one token at a time, producing a single context vector $c$ from its final hidden state.
2. **Decoder:** Takes $c$ as its initial hidden state and generates the output sequence one token at a time.

$$
\begin{array}{c|c}
\textbf{Encoder (input sentence)} & \textbf{Decoder (output sentence)} \\
\hline
x_1 \xrightarrow{W_{xh}} h_1 & \\
x_2 \xrightarrow{W_{xh}} h_2 & \\
x_3 \xrightarrow{W_{xh}} h_3 & \\
x_4 \xrightarrow{W_{xh}} h_4 & \\
\quad\downarrow & \\
c = h_4 & \xrightarrow{\text{init}} d_0 = c \\
\quad\downarrow & \quad\downarrow \\
& d_1 \to y_1 \\
& d_2 \xrightarrow{W_{xh}} h_2' \to y_2 \\
& d_3 \xrightarrow{W_{xh}} h_3' \to y_3
\end{array}
$$

#### The Information Bottleneck

The context vector `c` must encode the *entire meaning* of the input sentence. For a 50-word sentence, the encoder compresses 50 vectors into 1.

This is the **information bottleneck** of Seq2Seq:
- The decoder cannot easily "look back" at specific parts of the input
- Long sentences lose information
- Different output words might need to attend to *different* parts of the input

*Real-world example:* The first Google Translate neural system (2016) used a Seq2Seq LSTM model. Long sentences were translated poorly because of this bottleneck.

#### Attention — The Precursor to Transformers

**Bahdanau et al. (2015)** solved this by introducing *attention*: instead of compressing the input into one vector, the decoder looks at *all* of the encoder's hidden states, with a learned weighting.

$$
\begin{aligned}
c_t &= \sum_{i=1}^{T} \alpha_{t,i}\, h_i \\
\alpha_{t,i} &= \mathrm{softmax}\!\left(\mathrm{score}(d_{t-1},\, h_i)\right)
\end{aligned}
$$

At each decoding step $t$, the decoder computes a weighted sum of all encoder states, with weights that depend on the current decoder state. This was the critical insight that set the stage for Transformers.

---

### Lesson 2: Transformers — The Architecture That Changed Everything

*Real-world example:* On June 12, 2017, Google switched Google Translate to a Transformer-based system. Within weeks, users reported translations were *qualitatively different* — more natural, more idiomatic, more aware of context across long sentences. One architecture change displaced 30 years of RNN research.

#### Self-Attention Intuition

Read this sentence:  
*"The trophy didn't fit in the suitcase because **it** was too small."*

You, the human reader, know that "it" refers to the suitcase (because a small suitcase can't contain a large trophy). To figure this out, your brain pays attention to *every* word in the sentence and weighs their relevance to "it."

Self-attention formalizes this process. **Every word in the sequence computes a weighted combination of every other word, where the weights are learned dynamically based on the content.**

#### Queries, Keys, and Values — The Library Analogy

Imagine you're searching for a book in a library:

- **Query (Q):** What you're looking for. ("A book on quantum mechanics for beginners.")
- **Key (K):** What each book on the shelf *advertises* about itself. ("I'm a physics textbook," "I'm an advanced monograph," "I'm a popular science intro.")
- **Value (V):** The actual content of each book.

You compare your **query** against every book's **key**, get a relevance score for each, and then read a weighted combination of their **values** — mostly the books with high scores, almost ignoring the rest.

In self-attention:
- Each word produces a **query** (what information am I looking for?)
- Each word produces a **key** (what information do I contain?)
- Each word produces a **value** (what do I actually contribute?)

The output for word `i` is a weighted sum of all *values*, weighted by how well each word's *key* matches word `i`'s *query*.

#### Scaled Dot-Product Attention

For a sequence of tokens, pack their queries, keys, and values into matrices $Q$, $K$, $V$:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

Step by step:

1. **Compute similarity scores:** $Q K^{\top}$ is a matrix where entry $(i, j)$ is the dot product of query $i$ with key $j$ — how much does word $i$ want to attend to word $j$?
2. **Scale:** Divide by $\sqrt{d_k}$ (the square root of the key dimension). This prevents dot products from growing too large, which would push the softmax into regions of tiny gradients.
3. **Normalize:** Softmax along each row turns the scores into a probability distribution that sums to 1.
4. **Aggregate:** Multiply by $V$. The result for position $i$ is a weighted average of all value vectors, weighted by the attention probabilities.

**Pseudo-Python:**

```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.T                    # similarity
    scores = scores / np.sqrt(d_k)      # scale
    if mask is not None:
        scores = scores + mask          # mask out future positions
    weights = softmax(scores, axis=-1)  # normalize
    return weights @ V
```

#### Self-Attention as a Graph

$$
\begin{array}{rcl}
\text{Token } i & \xrightarrow{\;W_Q, W_K, W_V\;} & Q_i,\; K_i,\; V_i \quad (i = 1, 2, 3, \ldots, n) \\
\\
(Q_1, Q_2, \ldots, Q_n),\; (K_1, K_2, \ldots, K_n) & \xrightarrow{\;Q K^{\top}\;} & 
\text{Scores matrix } (n \times n) \\
& \xrightarrow{\;\mathrm{softmax}(\cdot\,/\,\sqrt{d_k})\;} & 
\text{Attention weights } \alpha_{ij} \\
& \xrightarrow{\;\times\,(V_1, V_2, \ldots, V_n)\;} & 
\boxed{\text{Output } = \alpha \cdot V}
\end{array}
$$

In self-attention, Q, K, and V all come from the *same* sequence. (In cross-attention between encoder and decoder, Q comes from the decoder and K, V from the encoder.)

#### Why Self-Attention Is Revolutionary

Compared to recurrence:

| Property | RNN | Self-Attention |
|---|---|---|
| Long-range dependencies | Lost after ~10 steps | Direct path between any two positions |
| Parallelism | Sequential (one token at a time) | Fully parallel (all tokens at once) |
| Compute per layer | O(sequence_length) | O(sequence_length²) |
| Path length between positions | O(sequence_length) | O(1) |

The O(n²) compute is a cost — self-attention is more expensive per layer than an RNN. But GPU parallelism wins: you can compute the attention matrix for a 512-token sequence in the time an RNN takes to process 512 sequential steps. The trade-off favors transformers in almost every modern regime.

*Real-world example:* The Transformer architecture is now the foundation of every major language model — GPT, Claude, Llama, Gemini — and has expanded to vision (ViT), audio (Whisper), biology (AlphaFold), and code (Copilot).

#### Multi-Head Attention

A single attention head captures one type of relationship. **Multi-head attention** runs $h$ parallel attention operations and concatenates their outputs:

$$
\begin{aligned}
\mathrm{MultiHead}(Q, K, V) &= \mathrm{Concat}(\mathrm{head}_1, \mathrm{head}_2, \ldots, \mathrm{head}_h)\, W_O \\
\mathrm{head}_i &= \mathrm{Attention}(Q W_{Q,i},\ K W_{K,i},\ V W_{V,i})
\end{aligned}
$$

Each head has its own learned $W_Q$, $W_K$, $W_V$ projections. Different heads learn to attend to different aspects:

$$
\begin{array}{rcl}
\text{Input } X & \xrightarrow{\;W_{Q,i},\; W_{K,i},\; W_{V,i}\; \text{per head } i\;} & 
\underbrace{\begin{array}{c} \text{Head}_1 \\ \text{Head}_2 \\ \vdots \\ \text{Head}_h \end{array}}_{h \text{ parallel heads}} \\[1em]
& \xrightarrow{\;\mathrm{Concat}(\cdot)\;} & \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) \\
& \xrightarrow{\;\times\,W_O\;} & \boxed{\text{Output}}
\end{array}
$$

Typical head counts: 8 (original Transformer), 16-32 (Llama 3 8B), 64+ (larger models).

#### Positional Encoding

Self-attention has a fundamental issue: it is **permutation-equivariant**. If you shuffle the input tokens, the output is shuffled the same way — the model has no idea which word came first.

**Positional encoding** adds information about position. The original paper uses fixed sinusoidal encodings:

$$
\begin{aligned}
PE_{(\mathrm{pos},\ 2i)}   &= \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right) \\
PE_{(\mathrm{pos},\ 2i+1)} &= \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
\end{aligned}
$$

Where $\mathrm{pos}$ is the position and $i$ is the dimension index. These create a unique "barcode" for every position, and crucially, the encoding for position $\mathrm{pos} + k$ can be expressed as a linear function of the encoding for position $\mathrm{pos}$ — which lets the model learn to attend *by relative offset*.

Modern models often use **rotary positional embeddings (RoPE)** (used in Llama) or **ALiBi** (used in BLOOM) — both work better for long sequences.

*Real-world example:* Without positional encoding, the sentences "the dog bit the man" and "the man bit the dog" would produce identical outputs from the self-attention layers, even though they have opposite meanings.

#### The Transformer Block

A transformer encoder block consists of:

1. **Multi-head self-attention**
2. **Add & Norm** (residual connection + layer normalization)
3. **Feed-forward network** (two linear layers with a nonlinearity)
4. **Add & Norm**

$$
\begin{aligned}
x' &= \mathrm{LayerNorm}\!\left(x + \mathrm{MultiHeadAttention}(x)\right) \\
y  &= \mathrm{LayerNorm}\!\left(x' + \mathrm{FFN}(x')\right) \\
\mathrm{FFN}(z) &= \max(0,\ z W_1 + b_1)\, W_2 + b_2
\end{aligned}
$$

The **residual connection** (the $x + \ldots$ part) is critical: it lets gradients flow backward through the network without vanishing, even when the attention or FFN produces near-zero outputs. This is what lets transformers scale to dozens or hundreds of layers.

#### The Full Transformer Architecture

$$
\begin{array}{c|c}
\textbf{Encoder (left)} & \textbf{Decoder (right)} \\
\hline
\text{Input tokens} & \\
\quad\downarrow & \\
\quad + \text{Positional encoding} & \\
\quad\downarrow & \\
\text{Encoder block } 1 & \\
\quad\downarrow & \\
\text{Encoder block } 2 & \\
\quad\downarrow & \\
\quad\vdots & \\
\quad\downarrow & \\
\text{Encoder block } N & \\
\quad\downarrow & \\
\text{Encoder output} \;\dashrightarrow\;\dashrightarrow\;\dashrightarrow & 
\begin{array}{c} \text{cross-attention} \\ \xleftarrow{\text{(queries from decoder, K/V from encoder)}} \end{array} \\
& \text{Output tokens (shifted right)} \\
& \quad\downarrow \\
& \quad + \text{Positional encoding} \\
& \quad\downarrow \\
& \text{Masked self-attention} \\
& \quad\downarrow \\
& \text{Cross-attention} \\
& \quad\downarrow \\
& \text{Feed-forward} \\
& \quad\downarrow \\
& \text{Decoder block } 2 \\
& \quad\downarrow \\
& \quad\vdots \\
& \quad\downarrow \\
& \text{Decoder block } N \\
& \quad\downarrow \\
& \text{Linear} + \text{Softmax} \\
& \quad\downarrow \\
& \boxed{\text{Output probabilities}}
\end{array}
$$

Key features:
- The encoder is **bidirectional**: every position can attend to every other (including future ones).
- The decoder uses **masked self-attention**: position `t` can only attend to positions `<= t`, preventing it from "cheating" by looking at the answer.
- The decoder also has **cross-attention** layers where the queries come from the decoder and the keys/values come from the encoder output.

#### Number of Layers in Real Models

| Model | Encoder layers | Decoder layers | Hidden dim | Heads |
|---|---|---|---|---|
| Original Transformer (2017) | 6 | 6 | 512 | 8 |
| Transformer Big | 6 | 6 | 1024 | 16 |
| BERT-base | 12 | — | 768 | 12 |
| BERT-large | 24 | — | 1024 | 16 |
| GPT-2 | — | 12 | 768 | 12 |
| GPT-3 | — | 96 | 12288 | 96 |
| Llama 3 70B | — | 80 | 8192 | 64 |
| Llama 3 405B | — | 126 | 16384 | 128 |

#### The Transformer Family — Three Variants

The original Transformer was designed for sequence-to-sequence tasks (translation). Three families of variants have emerged:

**Encoder-only (BERT family):**  
Bidirectional attention. Best for tasks that need full context understanding — classification, named entity recognition, sentence embeddings. Trained with masked language modeling (predict the masked word).

**Decoder-only (GPT family):**  
Masked (causal) attention. Best for text generation. Trained with next-token prediction. Powers ChatGPT, Claude, Llama, Mistral, Gemma.

**Encoder-decoder (T5, BART):**  
Full original architecture. Best for translation, summarization, and other transformation tasks where input and output are both sequences.

$$
\begin{array}{c}
\textbf{Original Transformer (2017)} \\
\downarrow \\
\left\{
\begin{array}{lll}
\textbf{Encoder-only} & \xrightarrow{\;W_{Q,i},\; W_{K,i},\; W_{V,i}\;} & \text{BERT, RoBERTa, DeBERTa} \\[0.4em]
\textbf{Decoder-only} & \xrightarrow{\;\text{masked self-attention}\;} & \text{GPT-2, GPT-3, GPT-4, Llama 2, Llama 3, Mistral} \\[0.4em]
\textbf{Encoder-decoder} & \xrightarrow{\;\text{full original architecture}\;} & \text{T5, FLAN-T5, BART}
\end{array}
\right.
\end{array}
$$

#### Transformer vs RNN — Summary

| Dimension | RNN / LSTM | Transformer |
|---|---|---|
| Sequence processing | Sequential (one token at a time) | Parallel (all tokens at once) |
| Long-range dependencies | Limited by gradient flow | Direct attention between any positions |
| Training time on GPU | Slow (cannot parallelize time) | Fast (full sequence in one forward pass) |
| Inference | Slow (one token at a time for generation) | Slow for generation (KV cache helps) |
| Memory for length n | O(n) state | O(n²) attention matrix |
| Best for | Streaming, very long sequences | Most practical NLP tasks today |

The single biggest practical advantage of the Transformer: **training speed**. The parallelism lets you train on thousands of GPUs simultaneously, which is what made trillion-parameter models feasible.

---

### Lesson 3: Transformers Deep Dive — Tokens, Attention & LLM Foundations

*Real-world example:* Between 2018 and 2023, the largest language models grew from 340M to 1.8T parameters — a 5,000× increase. Yet the *architecture* barely changed: both GPT-2 and GPT-4 are decoder-only transformers with masked self-attention. The differences are scale, data, compute, and training tricks.

#### The Objective: Next-Token Prediction

At its core, every LLM does one thing: given a sequence of tokens, predict the probability distribution over the next token.

$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

The model is trained to maximize the log-probability of the correct next token across an enormous corpus:

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t}) \quad \text{for all } t \text{ in the training corpus}
$$

This is called **language modeling**, and it is *autoregressive* — the output of one step is the input to the next.

Everything else — reasoning, planning, conversation, code generation — emerges from this one objective applied at sufficient scale. There is no separate "reasoning module" or "knowledge base." Just: predict the next token, very well, over very many tokens.

#### What Scales With Parameters

Empirically, more parameters → better performance on almost every metric, but with **diminishing returns**. Key scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022):

- **Kaplan (2020):** Performance scales as a power law with parameter count, dataset size, and compute — independently.
- **Chinchilla (2022):** For optimal compute, you should train on **~20 tokens per parameter**. A 70B model should be trained on ~1.4T tokens. Llama 3 70B was trained on 15.6T tokens — well past the Chinchilla point, with quality improvements from the extra data.

$$
\mathrm{Optimal\ tokens} \approx 20 \times \mathrm{parameters}
$$

| Model | Parameters | Training tokens | Tokens per param |
|---|---|---|---|
| GPT-2 | 1.5B | ~10B | ~7 (under-trained) |
| GPT-3 | 175B | ~300B | ~1.7 (under-trained) |
| LLaMA 1 65B | 65B | 1.4T | ~22 (Chinchilla-optimal) |
| Llama 2 70B | 70B | 2T | ~29 |
| Llama 3 70B | 70B | 15.6T | ~223 |

#### What the Parameters Encode

The billions (or trillions) of parameters collectively encode:
- **Knowledge** of the world: facts, dates, concepts, named entities
- **Patterns of language**: grammar, idioms, register, style
- **Reasoning patterns**: how to break down problems, chain steps
- **Code structure**: syntax, common algorithms, API patterns
- **Common sense**: physical reasoning, social norms, defaults

All of this is *implicit* in the weights, distributed across all the attention heads and FFN neurons. There is no clean separation — it is not the case that "neuron 1,234,567 stores the capital of France." Knowledge is spread across the entire network in superposition.

#### The Training Pipeline: Pre-training, SFT, RLHF

A modern LLM is trained in three distinct stages:

$$
\begin{array}{c|c|c}
\textbf{Stage 1: Pre-training} & \textbf{Stage 2: SFT} & \textbf{Stage 3: RLHF} \\
\hline
\begin{array}{c} \text{Raw internet text} \\ \text{(Common Crawl, books,} \\ \text{code, Wikipedia)} \end{array} & 
\begin{array}{c} \text{Curated instruction-response} \\ \text{pairs } (\sim 100K \text{ examples}) \end{array} & 
\begin{array}{c} \text{Human preference} \\ \text{rankings } (\sim 100K \\ \text{comparisons}) \end{array} \\
\downarrow & \downarrow & \downarrow \\
\text{Next-token prediction} & \text{Supervised Fine-Tuning} & \text{Reward model + PPO} \\
\text{(tens of millions USD)} & \text{Learn to follow instructions} & \text{(millions USD)} \\
& \text{(thousands USD)} & \\
\downarrow & \downarrow & \downarrow \\
\text{Stage 1 output} & \text{Stage 2 output} & \\
\downarrow & \downarrow & \\
\multicolumn{3}{c}{\boxed{\text{Aligned assistant model}}}
\end{array}
$$

**Stage 1 — Pre-training:**  
The model is trained on raw text (trillions of tokens from Common Crawl, Wikipedia, books, GitHub, arXiv). It learns to predict the next token. The result is a "base model" that is knowledgeable but not useful as a chatbot — if you ask it a question, it will *continue* the text rather than answer.

**Stage 2 — Supervised Fine-Tuning (SFT):**  
The base model is fine-tuned on a smaller dataset of high-quality `(instruction, response)` pairs written by humans. After this, the model learns to *respond* to instructions rather than just continue.

**Stage 3 — Reinforcement Learning from Human Feedback (RLHF):**  
Humans rank the model's outputs by quality. A **reward model** is trained to predict these rankings. The LLM is then fine-tuned using **Proximal Policy Optimization (PPO)** to maximize the reward model's score, with a KL-divergence penalty to keep it close to the SFT model (preventing reward hacking).

*Variants:* DPO (Direct Preference Optimization) skips the explicit reward model and trains directly on preferences. RLAIF uses AI-generated preferences instead of human ones.

#### The Cost Reality

Pre-training a frontier model today costs $50M–$500M in compute alone. SFT and RLHF are much cheaper (single-digit millions) but require enormous human effort in data labeling.

This is why only a handful of organizations train frontier models from scratch — and why the rest of the industry fine-tunes open-source base models or uses APIs.

*Real-world example:* Meta released Llama 3 in 2024 as open weights, allowing the broader industry to fine-tune for specific use cases (medical, legal, code) without the multi-million-dollar pre-training cost.

#### The Output Is a Probability Distribution

At each step, the model outputs a probability distribution over the *entire vocabulary* (50,000+ tokens for GPT-style models). The choice of which token to actually emit is controlled by **sampling parameters**.

**Greedy decoding:** Always pick the highest-probability token.  
*Problem:* Repetitive, boring, often gets stuck in loops.

#### Temperature

Temperature scales the logits (raw scores) before softmax:

$$
P(\mathrm{token}_i) = \frac{\exp(\mathrm{logit}_i\, /\, T)}{\sum_{j} \exp(\mathrm{logit}_j\, /\, T)}
$$

Effects:

- **$T = 0$:** Equivalent to greedy. Always pick the top token.
- **$T < 1$ (e.g., 0.7):** Sharpens the distribution. Top tokens become more likely. Output is more focused, predictable.
- **$T = 1$:** Standard sampling. Use the model's raw distribution.
- **$T > 1$ (e.g., 1.2):** Flattens the distribution. More diverse, more random, more risk of incoherence.

```
Logits: [2.0, 1.0, 0.5, 0.1]   (4 tokens)

T=0.1:  P = [0.999, 0.001, 0.000, 0.000]   (nearly deterministic)
T=1.0:  P = [0.534, 0.203, 0.121, 0.078]   (model's natural distribution)
T=2.0:  P = [0.355, 0.226, 0.176, 0.139]   (more uniform)
```

#### Top-K Sampling

Only sample from the K highest-probability tokens. Set the rest to zero probability before sampling.

```
K=1:    Greedy
K=5:    Sample from top 5 tokens
K=50:   Sample from top 50 tokens
K=vocab: No restriction
```

**Problem:** The right K depends on the distribution. A peaked distribution with K=50 might as well be K=3; a flat distribution with K=5 throws away most of the diversity.

#### Top-P (Nucleus) Sampling

Sample from the *smallest set of tokens whose cumulative probability exceeds P*. Adapts dynamically to the distribution.

```
Sorted probs: [0.4, 0.3, 0.15, 0.10, 0.05]
P=0.8:   Take tokens with cumulative 0.4+0.3+0.15 = 0.85 ≥ 0.8 → use first 3 tokens
P=0.5:   Use first 2 tokens (cumulative 0.7 ≥ 0.5)
P=0.95:  Use first 4 tokens (cumulative 0.95)
```

Top-P is almost always better than fixed top-K because it adapts.

#### Sampling Cheat Sheet

| Task | Temperature | Top-P | Notes |
|---|---|---|---|
| Factual Q&A | 0.0–0.2 | — | Greedy or near-greedy |
| Code generation | 0.0–0.3 | 0.95 | Deterministic, correct |
| Data extraction | 0.0 | — | Greedy for consistency |
| Email / professional writing | 0.4–0.6 | 0.9 | Polished but not robotic |
| Creative writing | 0.7–1.0 | 0.95 | More variation |
| Brainstorming | 1.0–1.3 | 0.95+ | Maximum diversity |

The combination that almost always works for general assistants: **temperature 0.7, top-P 0.9**.

#### Why LLMs Hallucinate

A model "hallucinates" when it confidently states something false. Four root causes:

1. **Training data noise:** The internet contains misinformation. The model learns to reproduce it.
2. **No verification mechanism:** The model has no way to check whether what it's saying is true. It just predicts plausible continuations.
3. **Interpolation in embedding space:** Rare or specific facts get "smoothed out" — the model returns the *average* of similar facts, which may be wrong.
4. **Sycophancy from RLHF:** The model is rewarded for being confident and helpful, which biases it toward giving an answer even when it shouldn't.

**Mitigations:**
- **RAG (Retrieval-Augmented Generation):** Provide the model with retrieved documents at inference time, so it can ground its answers in real text.
- **Tool use:** Let the model call search engines, calculators, or databases to verify facts.
- **Confidence calibration:** Train the model to say "I'm not sure" when appropriate.
- **Better RLHF data:** Reward honest uncertainty rather than confident guesses.

*Real-world example:* A medical chatbot that hallucinates drug interactions can be life-threatening. RAG over a verified drug database is the standard mitigation in healthcare AI.

#### The Scaling Wall

A 2024-era frontier model might have 1.8T parameters. Could we go to 100T? There are reasons to think not:

- **Data:** We are running out of high-quality text. Common Crawl has roughly 100T tokens of useful content. Synthetic data (generated by other models) is being used but has quality limits.
- **Compute:** Training compute has grown ~5× per year. We are approaching the limits of what a single datacenter can hold.
- **Diminishing returns:** Each doubling of parameters gives a smaller improvement than the last. Scaling alone is not enough.

This is why the field is moving toward:
- **Smaller, better models** (Llama 3 8B outperforms GPT-3 175B on many tasks)
- **Mixture of Experts (MoE)** to scale parameters without scaling compute proportionally
- **Better data, not just more data** (curation, synthetic data, multi-modal training)
- **Inference-time compute** (let the model think longer for harder questions, as in o1/o3)

#### Mixture of Experts (MoE)

The trick to trillion-parameter models without trillion-parameter compute: **only use a fraction of the network for any given token**.

$$
\begin{array}{c}
\text{Input token} \\
\downarrow \\
\text{Router network} \\
\downarrow \\
\left\{
\begin{array}{l}
\text{Expert 1} \\
\text{Expert 2} \\
\text{Expert 3} \\
\text{Expert 4} \\
\text{Expert 5} \\
\text{Expert 6} \\
\text{Expert 7} \\
\text{Expert 8}
\end{array}
\right.
\xrightarrow{\;\text{top-}k\;\text{routing}\;}
\underbrace{\sum_{i \in \text{top-}k} w_i \cdot \text{Expert}_i}_{\text{weighted sum}}
\xrightarrow{\;\;\;}
\boxed{\text{Output}}
\end{array}
$$

Each MoE layer has N "expert" sub-networks (e.g., 8 or 64). A small **router** network decides which 2 (or so) experts to use for each token. The output is a weighted combination of those experts.

**Why it works:**  
Total parameters: 1.8T. Compute per token: roughly equivalent to a 200B dense model. Cost grows much more slowly than parameter count.

**Used in:** Mixtral 8x7B, GPT-4 (rumored), Gemini 1.5, Llama 4 (rumored).

#### Model Architecture Comparison

| Model | Year | Type | Layers | Hidden | Heads | Params | Context |
|---|---|---|---|---|---|---|---|
| GPT-2 | 2019 | Decoder | 12–48 | 768–1600 | 12–25 | 117M–1.5B | 1024 |
| GPT-3 | 2020 | Decoder | 96 | 12288 | 96 | 175B | 2048 |
| GPT-3.5 | 2022 | Decoder | 96+ | 12288+ | 96+ | ~175B | 16K |
| GPT-4 | 2023 | Decoder (MoE) | 120+ | — | — | ~1.8T (rumored) | 128K |
| Llama 2 70B | 2023 | Decoder | 80 | 8192 | 64 | 70B | 4K |
| Llama 3 70B | 2024 | Decoder | 80 | 8192 | 64 | 70B | 128K |
| Llama 3.1 405B | 2024 | Decoder | 126 | 16384 | 128 | 405B | 128K |
| Claude 3.5 Sonnet | 2024 | Decoder (rumored) | — | — | — | ~200B+ | 200K |
| Gemini 1.5 Pro | 2024 | Decoder (MoE) | — | — | — | — | 1M–2M |

The trend: bigger context windows (now routinely 128K–2M tokens), bigger parameter counts via MoE, and architecture diversity (some models use sliding-window attention, some use mixture of depths, etc.).

---

### Key Takeaways — Sequence Models and Transformers

1. **Sequences require models that explicitly handle order**; feedforward networks treat inputs as unordered bags.
2. **Vanilla RNNs** maintain a hidden state that is updated at each time step using the same weights — but the vanishing gradient problem prevents them from learning dependencies more than ~10 steps apart.
3. **LSTMs and GRUs** solve this with gating mechanisms: LSTMs use three gates plus a cell state highway, GRUs use two gates. Both preserve information across long sequences.
4. **Seq2Seq** uses an encoder-decoder pattern but suffers from an information bottleneck when compressing the input into one vector.
5. **Attention (2015)** let the decoder look back at *all* encoder states — the direct precursor to the Transformer.
6. **Self-attention** lets every token in a sequence look at every other token and dynamically weight their contributions via Query/Key/Value projections.
7. **Multi-head attention** runs multiple attention operations in parallel; different heads learn different relationship types.
8. **Positional encoding** (sinusoidal, RoPE, or ALiBi) injects order information that pure self-attention lacks.
9. **Residual connections** are essential for training deep transformer stacks without vanishing gradients.
10. **The transformer family has three branches:** encoder-only (BERT), decoder-only (GPT/Llama), and encoder-decoder (T5).
11. **Modern LLMs are trained in three stages:** pre-training (raw text), SFT (instruction-following), and RLHF (alignment with human preferences).
12. **Temperature, top-K, and top-P** control how the model samples from its output distribution — temperature for "sharpness," top-K and top-P for "diversity bounds."
13. **Hallucination** has structural causes (no verification, training data noise) and is mitigated by RAG, tool use, and better data — not by scaling alone.
14. **Mixture of Experts (MoE)** lets models reach trillion parameters while keeping per-token compute manageable.

#### The Evolution of Sequence Models

The journey from RNNs to modern LLMs:

$$
\begin{array}{c}
\text{Boolean Retrieval (1950s--1990s)} \\
\quad\downarrow\;\text{keyword matching} \\
\text{Bag of Words / TF-IDF (2000s)} \\
\quad\downarrow\;\text{no semantics, no order} \\
\text{Word Embeddings (2013)} \\
\quad\downarrow\;\text{meaning, but still no order} \\
\text{RNNs / LSTMs (2014--2017)} \\
\quad\downarrow\;\text{order, but slow and limited range} \\
\text{Seq2Seq + Attention (2015)} \\
\quad\downarrow\;\text{dynamic focus on context} \\
\text{Transformers (2017)} \\
\quad\downarrow\;\text{parallel, scalable, attention is all you need} \\
\text{BERT / GPT (2018+)} \\
\quad\downarrow\;\text{pretrained on massive text, fine-tuned for tasks} \\
\text{LLMs with RLHF (2020+)} \\
\quad\downarrow\;\text{emergent reasoning, instruction following} \\
\text{Mixture of Experts (2023+)} \\
\quad\downarrow\;\text{trillion parameters at fixed compute} \\
\boxed{\text{Inference-time reasoning (o1/o3, 2024+)}} \\
\quad\downarrow\;\text{let the model think longer for harder questions}
\end{array}
$$

> The Transformer architecture is the foundation of every modern AI system — from chatbots to code copilots to scientific discovery tools. Understanding *why* it works (attention, parallel processing, gating) is more important than memorizing the formulas.
