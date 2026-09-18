# Generative AI with LSTM - Text Generation

An end-to-end Deep Learning model built with TensorFlow/Keras using Long Short-Term Memory (LSTM) networks to generate stylized text sequences based on initial seed phrases.

---

## 1. Dataset & Preprocessing
* **Dataset**: Shakespeare's Plays and Sonnets (Tiny Shakespeare corpus).
* **Source**: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
* **Download Instructions**: The dataset is downloaded automatically inside `main.py` via `tf.keras.utils.get_file`.
* **Preprocessing Pipeline**:
  * Stripped all punctuation and converted text to lowercase.
  * Tokenized sentences into word-level indices using Keras `Tokenizer` (Vocabulary size: 1,617 unique tokens).
  * Generated cumulative $N$-gram sequences matching sequential next-word prediction targets.
  * Applied pre-padding with `pad_sequences` (Maximum sequence length: 12).
  * Partitioned dataset into an 80% training set (4,173 samples) and 20% validation set (1,044 samples).

---

## 2. Model Architecture & Compilation
* **Embedding Layer**: Projects token indices into a dense 100-dimensional vector space.
* **LSTM Layer**: 128 units with recurrent dropout (0.2) to maintain sequential dependencies and mitigate overfitting.
* **Dense Layer**: 128 units with ReLU non-linear activation.
* **Regularization Layer**: Dropout (0.2) to regularize hidden feature maps.
* **Output Layer**: Dense Softmax layer over the complete 1,617 vocabulary classes.
* **Compilation**: Optimized using Adam ($lr = 0.005$) and evaluated via Categorical Crossentropy loss.
* **Callbacks**: Configured `EarlyStopping(patience=6)` and `ModelCheckpoint` saving the optimal model state to `best_lstm_model.keras`.

---

## 3. Generated Sample Outputs
Generated using probabilistic temperature sampling ($T = 0.75$):

* **Prompt 1**: `"to be or not to be"`
  > *Generated*: `to be or not to be receive the hare up it must you have been mammocked done for us thou that`

* **Prompt 2**: `"shall i compare thee"`
  > *Generated*: `shall i compare thee show fear us we have little not a fawning greyhound o the purpose i is`

* **Prompt 3**: `"the king has spoken"`
  > *Generated*: `the king has spoken we have been he bearing you not about ten true corioli valiant titus lartius not`

---

## 4. Bonus: Architecture & Hyperparameter Exploration
* **Single vs. Deep Stacked LSTM**: Adding a secondary recurrent layer (`LSTM(64)`) improves syntactic representation over complex clauses, but adds significant memory overhead and requires stricter dropout ($0.3$) to avoid gradient saturation.
* **Sequence Length Sensitivity**: Shorter sequences ($N \le 6$) lead to faster training but cause repetitive lexical loops; sequences around $N = 12$ capture phrase-level Shakespearean cadence while retaining stable runtime.
* **Temperature Tuning**: A temperature parameter of $T = 0.75$ yields the ideal balance between strict grammatical adherence and novel vocabulary generation.

---

## 5. How to Run
```bash
# Clone the repository
git clone <your-repo-link>
cd lstm-text-generation

# Install dependencies
pip install tensorflow numpy

# Execute script
python main.py
```