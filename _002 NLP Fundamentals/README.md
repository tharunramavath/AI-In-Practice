# NLP Fundamentals

Welcome to the foundational module on Natural Language Processing. This folder introduces the core concepts, techniques, and pipelines used to process and understand human language with computers.

---

### Lesson 1: What is NLP?

**Natural Language Processing (NLP)** is a subfield of Artificial Intelligence that focuses on the interaction between computers and human language. The goal is to enable computers to read, understand, interpret, and generate text the way humans do.

#### Why NLP is Hard

Human language is:

- **Ambiguous** — "I saw the man with the telescope." (Who has the telescope?)
- **Context-dependent** — "It's cold in here" might mean "close the window."
- **Constantly evolving** — new slang, idioms, emojis
- **Culturally embedded** — "break a leg" doesn't mean what it literally says

*Real-world example:* A customer support chatbot must understand "my order hasn't shown up yet," "where's my pkg?," and "I never received it bro" — all meaning the same thing.

#### Real-World Applications of NLP

| Domain | Example Application |
|--------|---------------------|
| Search | Google, Bing ranking and query understanding |
| Assistants | Siri, Alexa, Google Assistant |
| Translation | Google Translate, DeepL |
| Sentiment | Brand monitoring, review analysis |
| Healthcare | Clinical note parsing, medical coding |
| Finance | Earnings call analysis, fraud detection |
| Legal | Contract review, e-discovery |
| Code | GitHub Copilot, code review tools |

#### The NLP Pipeline

A typical NLP system processes text through a series of stages:

```
Raw Text
   ↓
Tokenization
   ↓
Normalization (lowercase, remove noise)
   ↓
Stop Word Removal (optional in modern DL)
   ↓
Stemming / Lemmatization (optional in modern DL)
   ↓
Feature Extraction (BoW / TF-IDF / Embeddings)
   ↓
Model (Classification / Generation / Translation)
   ↓
Output
```

- Early NLP systems followed this pipeline rigidly with hand-crafted features.
- Modern deep learning systems learn the entire pipeline end-to-end — but the conceptual stages still apply.

---

### Lesson 2: Text Preprocessing

Preprocessing converts raw, messy text into a clean, structured form that models can consume.

#### Tokenization

**Definition:** Breaking text into smaller units called *tokens* — usually words or sub-words.

```
Input:  "I love deep learning!"
Tokens: ["I", "love", "deep", "learning", "!"]
```

- Tokens are the **LEGO bricks** of NLP — you break down a sentence into individual bricks before building anything.
- *Technical note:* Modern tokenizers like BPE (Byte Pair Encoding) break words into sub-word units: `"unbelievable"` might become `["un", "believ", "able"]`. We'll cover BPE in depth in a later module on LLMs and tokenization.

*Real-world example:* Search engines tokenize queries to match against indexed documents.

#### Stop Word Removal

**Definition:** Removing common words (like "the," "is," "at") that add little meaning.

```
Before: "The cat sat on the mat"
After:  ["cat", "sat", "mat"]
```

- This was crucial in the early keyword-matching days.
- Today's deep learning models *don't* remove stop words — they learn their importance contextually.

*Real-world example:* TF-IDF-based spam filters ignore stop words because they appear in every email.

#### Stemming vs Lemmatization

| Concept | Definition | Example |
|---------|------------|---------|
| **Stemming** | Chop word endings off (crude, fast) | "running" → "run", "studies" → "studi" |
| **Lemmatization** | Find the dictionary root (smart, slow) | "running" → "run", "better" → "good" |

- **Stemming** is like using a chainsaw — fast, but leaves rough edges.
- **Lemmatization** is like using a scalpel — precise, uses actual grammar rules.

*Real-world example:* Search engines use stemming so that "running shoes" matches documents containing "run shoes" or "runs shoes."

#### Normalization & Noise Cleaning

Common normalization steps:

- **Lowercasing** — `"Hello"` → `"hello"`
- **Removing punctuation** — `"Wait!"` → `"Wait"`
- **Removing numbers** (sometimes) — `"Order 123"` → `"Order"`
- **Removing special characters** — `"café"` → `"cafe"`
- **Handling contractions** — `"don't"` → `"do not"`
- **Removing HTML tags / URLs** — for web-scraped text

*Real-world example:* A Twitter sentiment analyzer strips mentions, hashtags, URLs, and emojis before classifying the actual sentiment of the tweet.

---

### Lesson 3: Linguistic Features

These techniques add structured, grammatical understanding to raw text.

#### Part-of-Speech (POS) Tagging

**Definition:** Labeling each word with its grammatical role (noun, verb, adjective, etc.).

```
"The quick brown fox jumps over the lazy dog"
 DET  ADJ   ADJ   NN   VBZ  IN  DET  ADJ   NN
```

- **Why it matters:** "Apple" in "Apple CEO Tim Cook" (noun) vs. "apple pie" (adjective modifier) behaves differently in a sentence.
- POS tagging helps disambiguate words that look the same but play different roles.

*Real-world example:* Grammar checkers like Grammarly use POS tagging to detect incorrect verb tenses or subject-verb agreement errors.

#### Named Entity Recognition (NER)

**Definition:** Identifying and categorizing named entities — people, organizations, locations, dates.

```
"Elon Musk founded Tesla in 2003 in California."
 PERSON         ORG       DATE    LOCATION
```

Common entity categories:

- **PERSON** — names of people
- **ORG** — companies, institutions
- **GPE** — countries, cities, states
- **DATE / TIME** — absolute or relative dates
- **MONEY / PERCENT** — financial values
- **PRODUCT** — consumer products

- Used heavily in: financial news parsing, legal document review, medical record analysis.

*Real-world example:* Financial news parsers extract company names and earnings figures from articles to feed trading algorithms.

#### Dependency Parsing

**Definition:** Analyzing the grammatical structure of a sentence to find relationships between words.

- Identifies the **subject**, **object**, and **modifiers** of each verb.
- Produces a parse tree showing how words depend on one another.

*Real-world example:* Used in question-answering systems to identify "who did what to whom" in a sentence.

---

### Lesson 4: Classical Text Representation

Before deep learning, text had to be converted into numbers using hand-crafted methods.

#### Bag of Words (BoW)

**Definition:** Represent text as a collection (bag) of words, ignoring order and grammar.

```
Sentence A: "The dog bit the man"
Sentence B: "The man bit the dog"

BoW representation:
{"the": 2, "dog": 1, "bit": 1, "man": 1}  ← SAME for both!
```

- The "bag" is just a count of how many times each word appears.

**The problem:** "The dog bit the man" and "The man bit the dog" have *identical* BoW representations but *opposite* meanings.

*This is why we needed better models. Enter word vectors.*

*Real-world example:* Early spam filters used BoW features to classify emails based on word frequency.

#### TF-IDF — Smarter than Bag of Words

**TF-IDF** stands for **Term Frequency – Inverse Document Frequency**.

- **TF (Term Frequency):** How often does word W appear in document D?
- **IDF (Inverse Document Frequency):** How rare is word W across *all* documents?

**Formula:**

```
TF-IDF(w, d) = TF(w, d) × log(N / df(w))

Where:
  N    = total number of documents
  df(w) = number of documents containing word w
```

**Why this works:** The word "the" appears in every document, so it has a high TF but near-zero IDF — its TF-IDF score is low. The word "neutrino" is rare — high IDF — so if it appears in a document, it's very significant.

- Used heavily in: **search engines, document ranking, spam filters**.

*Real-world example:* Elasticsearch and Lucene use TF-IDF as the foundation of their ranking algorithms.

#### Limitations of Classical NLP

| Limitation | Explanation |
|------------|-------------|
| **Sparsity** | Vectors are huge (one entry per word in vocabulary) but mostly zeros |
| **No semantics** | "happy" and "joyful" are treated as completely unrelated |
| **No order** | Word order is lost in BoW / TF-IDF |
| **No generalization** | New words not in training vocab are invisible to the model |
| **Hand-crafted** | Features must be designed by humans, not learned from data |

*Real-world example:* Classical search engines could not understand that "car" and "automobile" are the same concept — that's why we needed embeddings.

---

### Lesson 5: Word Embeddings

Embeddings solved the biggest problem of classical NLP: **giving words actual meaning in numerical form**.

#### What are Word Embeddings?

**The Big Idea:** What if we could represent a word not as a string of letters, but as a **point in multi-dimensional space**, where similar words cluster together?

- Each word is mapped to a dense vector (e.g., 300 numbers).
- Words with similar *meanings* are close in **vector space**.

#### Word2Vec

**Word2Vec (2013, Google)** was trained on billions of words to learn that:

- "King" − "Man" + "Woman" ≈ "Queen"
- "Paris" − "France" + "Italy" ≈ "Rome"

**Architecture:** Word2Vec is a **shallow 2-layer neural network** — input layer, projection layer, output layer, with no hidden layer. Despite its simplicity, it produces remarkably powerful embeddings.

**Training Process:**

- Define a **context window** (typically 5–10 words on each side of a target word).
- Slide this window across the entire training corpus.
- Train the network to either predict the target word from context (CBOW) or predict context from target (Skip-gram).
- After training, the weights of the projection layer **become** the word embeddings.

**Two training approaches:**

- **CBOW (Continuous Bag of Words):** Predict the target word from surrounding context. Faster to train and works slightly better with frequent words.
- **Skip-gram:** Predict surrounding context from the target word. Slower but works better with rare words and tends to produce better embeddings for infrequent terms.

**Example for each:**

Sentence: *"The quick brown fox jumps over the lazy dog"*
Target word: **"fox"** with a context window of 2

- **CBOW (context → target):**
  - Input: `["quick", "brown", "jumps", "over"]` (the 2 words on each side of "fox")
  - Output: `"fox"`
  - The model sees the context words and learns to predict the missing center word.

- **Skip-gram (target → context):**
  - Input: `"fox"`
  - Output: Predict each context word separately — `"quick"`, `"brown"`, `"jumps"`, `"over"`
  - The model sees the target word and learns to predict every word that surrounds it (one at a time).

- The key difference: **CBOW** averages information from many context words to predict one word (smooths out noise, good for frequent words), while **Skip-gram** uses one word to predict many (preserves more nuance, better for rare words).

**Key Training Tricks:**

- **Negative Sampling:** Instead of updating all vocabulary weights for every training example (computationally prohibitive), only update a small number of "negative" (random) words along with the positive example. This makes training **1000x+ faster** with minimal quality loss.
- **Subsampling of Frequent Words:** Words like "the", "a", "is" appear so often they don't add much signal. Word2Vec randomly drops them — the more frequent the word, the higher the chance of being dropped. This dramatically improves embeddings for rare words.

**Why it works:** Words that appear in similar contexts end up with similar embeddings. The model implicitly learns that "cat" and "dog" are similar because they appear in similar sentences (e.g., "The ___ sat on the mat," "My ___ loves to play").

*Real-world example:* Google's original Word2Vec model was trained on 100 billion words from Google News, producing 300-dimensional embeddings for 3 million words and phrases. It became the foundation of countless downstream NLP applications.

#### GloVe

**GloVe (Global Vectors, Stanford 2014)** improved Word2Vec by training on **global word co-occurrence statistics** across the entire corpus, instead of just local context windows.

**The Core Insight:** Word2Vec only uses local context windows — it doesn't directly use the global statistics of the corpus. GloVe combines the best of both worlds: **local context (like Word2Vec)** + **global statistics (like older methods such as LSA — Latent Semantic Analysis)**.

**How GloVe Works:**

1. **Build a co-occurrence matrix X:** For every word pair (i, j), count how many times they appear within a context window of each other in the entire corpus. X_ij = number of times word j appears in the context of word i.
2. **The key insight:** **Ratios of co-occurrence probabilities**, not raw counts, encode meaning.
3. **Train embeddings** that satisfy a specific objective function designed to capture these ratios.

**The Intuition Behind Ratios:**

Consider the words *ice*, *steam*, *solid*, *gas*, *water*, *fashion*:

| Probability | Value | Meaning |
|-------------|-------|---------|
| P(solid \| ice) | High | Ice is solid |
| P(solid \| steam) | Low | Steam is not solid |
| P(gas \| ice) | Low | Ice is not gaseous |
| P(gas \| steam) | High | Steam is gaseous |
| **P(solid \| ice) / P(solid \| steam)** | **Very high** | Discriminates ice from steam |
| **P(fashion \| ice) / P(fashion \| steam)** | **≈ 1** | Fashion is unrelated to both — no discrimination |

These ratios cleanly separate relevant words from irrelevant ones — and GloVe's objective function is designed to make the dot product of two word vectors equal to the log of their co-occurrence probability.

**The GloVe Objective Function:**

```
J = Σ f(X_ij) × (w_i · w_j + b_i + b_j − log(X_ij))²

Where:
  X_ij    = number of times word j appears in the context of word i
  w_i, w_j = word vectors for words i and j (what we want to learn)
  b_i, b_j = bias terms
  f        = weighting function that prevents rare co-occurrences
             from dominating (commonly f(x) = (x/x_max)^α if x < x_max)
  ·        = dot product
```

- The model tries to make `w_i · w_j + b_i + b_j` ≈ `log(X_ij)` for all word pairs.

**Advantages of GloVe over Word2Vec:**

- **Faster training:** Leverages global statistics from the start, so the model converges much quicker.
- **Better performance** on word similarity and analogy benchmarks in many evaluations.
- **More interpretable:** The objective function has a clear probabilistic interpretation rooted in co-occurrence ratios.
- **Scales well:** Works efficiently on very large corpora.

*Real-world example:* Stanford's pre-trained GloVe vectors come in multiple sizes (50d, 100d, 200d, 300d) trained on Wikipedia, Common Crawl, and Twitter. They became a popular drop-in replacement for Word2Vec and are still used today as initialization for many NLP models.

#### The Magic of Vector Math

Word embeddings unlock **arithmetic on meaning**:

```
Word         Vector (simplified 3D)
-----------  ----------------------
King       → [0.9, 0.1, 0.8]
Queen      → [0.8, 0.9, 0.8]
Man        → [0.9, 0.1, 0.2]
Woman      → [0.8, 0.9, 0.2]
Apple      → [0.1, 0.2, 0.9]
```

- `King - Man + Woman ≈ Queen`
- This is not magic — it emerges from training on massive text where these words appear in similar contexts.

> **Key insight:** Embeddings are *learned*, not hand-crafted. The model figures out relationships purely from patterns in text data.

##### How a Word or Sentence Actually Becomes a Vector

The vector table above shows what embeddings *look like* — but how do we actually *get* them from real text? The process has two stages: **word → vector**, then **sentence → vector**.

**From Word to Vector (Lookup Table):**

The conversion from a word to its vector is just a **table lookup**. Models use a learned **embedding matrix** (also called an **embedding layer**):

1. Start with a vocabulary — say 50,000 words.
2. Initialize a matrix **E** of shape `(50000, 300)` — one 300-dim vector per word.
3. Each row of this matrix is the vector for one specific word.
4. The matrix **E** is **learned during training** — this is the embedding the model produces.
5. To convert the word `"king"` to a vector:
   - Convert `"king"` to its integer ID (e.g., `1234`)
   - Look up row `1234` in **E**
   - That row **IS** the vector for "king"

```
Vocabulary  →  Integer ID  →  Embedding Lookup  →  Vector
"king"     →     1234     →      E[1234]       →  [0.9, 0.1, 0.8, ...]
```

This is just a lookup — no computation is needed at inference time. The "intelligence" lives in **E**, which was learned from data.

**From Sentence to Vector:**

A sentence is just a **sequence of word vectors**. To get one vector for the whole sentence, the word vectors must be combined. The method you choose determines what information is preserved:

| Method | How It Works | Pros | Cons |
|--------|--------------|------|------|
| **Average / Sum** | Mean (or sum) of all word vectors in the sentence | Fast, simple, surprisingly strong baseline | Loses word order; "dog bites man" = "man bites dog" |
| **Concatenation** | Stack all word vectors into one long vector | Preserves every word's information | Only works for fixed-length sentences |
| **RNN / LSTM** | Process words one at a time, final hidden state = sentence vector | Captures word order and short context | Slow, suffers from long-range loss |
| **Transformer / Self-Attention** | Every word attends to every other word, in parallel | Captures order + long-range context, parallelizable | More compute, more complex |
| **Sentence-BERT / Universal Sentence Encoder** | Specifically trained to map sentences to a single vector | Plug-and-play semantic similarity | Requires a specialized pretrained model |

**Example: From Sentence to Vector**

Take the sentence: *"The king rules the kingdom."*

| Step | Operation | Result |
|------|-----------|--------|
| 1. Tokenize | Split into words | `["The", "king", "rules", "the", "kingdom"]` |
| 2. Convert to IDs | Map each word to its vocabulary ID | `[5, 1234, 5678, 5, 9012]` |
| 3. Look up embeddings | Pull each word's vector from **E** | Five 300-dim vectors |
| 4. Combine | Average / RNN / Transformer | One 300-dim sentence vector |

After step 3, the model has five 300-dim vectors — one per word. Step 4 then collapses them into a single 300-dim vector representing the entire sentence.

*Real-world example:* A sentiment classifier using **average embeddings** treats *"The movie was good"* and *"Good was the movie"* as **identical** — because order is lost in the average. A **Transformer-based** model correctly distinguishes them. This is why the choice of combination method matters as much as the choice of word vectors themselves.

*Real-world example:* Recommendation systems use embeddings to suggest "customers who liked X also liked Y" by finding similar items in embedding space.

---

### Key Takeaways — NLP Basics

1. **NLP is the science of teaching machines to understand human language**
2. **Language is deeply ambiguous** — machines need context, not just words
3. **Preprocessing** (tokenization, normalization, lemmatization) turns messy text into clean inputs
4. **Linguistic features** (POS, NER, dependency parsing) add structured meaning
5. **Classical representations** (BoW, TF-IDF) are simple but limited — they ignore order and semantics
6. **Word embeddings** (Word2Vec, GloVe) were the first breakthrough — giving words numerical meaning
7. All modern AI builds on these foundations — from chatbots to agents to code copilots

#### The Evolution of NLP

The journey from keyword matching to modern AI:

```
Bag of Words (1950s–2000s)
   ↓ ignores order, no semantics
Word Embeddings (2013)
   ↓ adds meaning, still no order
RNNs / LSTMs (2014–2017)
   ↓ adds order, slow to train
Seq2Seq + Attention (2015)
   ↓ dynamic focus on context
Transformers (2017)
   ↓ parallel, scalable, attention is all you need
BERT / GPT (2018+)
   ↓ pretrained on massive text, fine-tuned for tasks
Modern LLMs (2020+)
   ↓ emergent reasoning, instruction following, agents
```

> The Transformer architecture (covered in detail in the next module on sequence models) is the foundation of every modern AI system — and it all started with the simple idea of **teaching computers to read**.
