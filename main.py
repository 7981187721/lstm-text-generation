"""
Generative AI with LSTM - Text Generation
Dataset: Shakespeare's Text Corpus (Automated Download via tf.keras.utils)
Author: Lutukurti VenuGopal
"""

import os

# Suppress TensorFlow logging & oneDNN optimizations for clean console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Set deterministic seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ==========================================
# 1. Dataset Loading and Preprocessing
# ==========================================

# Step 1.1: Download the public domain Shakespeare corpus
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dataset_path = tf.keras.utils.get_file("shakespeare.txt", url)

with open(dataset_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Select 35,000 characters for robust vocabulary and rapid CPU training
corpus = raw_text[:35000]

# Step 1.2: Clean text: lowercase and strip punctuation
clean_corpus = corpus.lower().translate(
    str.maketrans("", "", string.punctuation)
)
text_lines = [
    line.strip() for line in clean_corpus.split("\n") if len(line.strip()) > 0
]

# Step 1.3: Tokenize words into integer IDs
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_lines)
vocab_size = len(tokenizer.word_index) + 1  # Reserve 0 for sequence padding

# Step 1.4: Generate n-gram sequences
input_sequences = []
for line in text_lines:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[: i + 1]
        input_sequences.append(n_gram_seq)

# Pad sequences to match the maximum line length
max_sequence_len = max(len(seq) for seq in input_sequences)
padded_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
)

# Step 1.5: Split into inputs (X) and target outputs (y)
X = padded_sequences[:, :-1]
y = tf.keras.utils.to_categorical(
    padded_sequences[:, -1], num_classes=vocab_size
)

# Step 1.6: Train / Validation split (80% train, 20% validation)
split_idx = int(0.8 * len(X))
indices = np.arange(len(X))
np.random.shuffle(indices)

X, y = X[indices], y[indices]
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Vocabulary Size: {vocab_size}")
print(f"Max Sequence Length: {max_sequence_len}")
print(f"Training Samples: {len(X_train)} | Validation Samples: {len(X_val)}")


# ==========================================
# 2. Model Architecture & Compilation
# ==========================================
def build_lstm_model(
    vocab_size, max_seq_len, embedding_dim=100, lstm_units=128, deeper=False
):
    model = Sequential()
    # Keras 3 standard: input_shape specifies sequence length
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_shape=(max_seq_len - 1,),
        )
    )

    if deeper:
        model.add(
            LSTM(lstm_units, return_sequences=True, dropout=0.2)
        )  # Stacked layer
        model.add(LSTM(lstm_units // 2, dropout=0.2))
    else:
        model.add(LSTM(lstm_units, dropout=0.2))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        metrics=["accuracy"],
    )
    return model


model = build_lstm_model(
    vocab_size, max_sequence_len, embedding_dim=100, lstm_units=128
)
model.summary()


# ==========================================
# 3. Model Training & Callbacks
# ==========================================
# Monitored on training loss with patience=6 to ensure the model trains deeply
callbacks = [
    EarlyStopping(
        monitor="loss", patience=6, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        "best_lstm_model.keras", monitor="loss", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)


# ==========================================
# 4. Text Generation Function
# ==========================================
def generate_text(seed_text, next_words, model, max_seq_len, temperature=0.8):
    """Generates continuous text using temperature-based probabilistic sampling."""
    output_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_seq_len - 1, padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)[0]

        # Apply temperature scaling to logits
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        probs = exp_preds / np.sum(exp_preds)
        predicted_idx = np.random.choice(len(probs), p=probs)

        predicted_word = tokenizer.index_word.get(predicted_idx, "")
        if not predicted_word:
            break
        output_text += " " + predicted_word

    return output_text


# ==========================================
# 5. Generated Sample Outputs
# ==========================================
seed_prompts = [
    "to be or not to be",
    "shall i compare thee",
    "the king has spoken",
]

print("\n" + "=" * 40)
print("       SAMPLE TEXT GENERATIONS       ")
print("=" * 40)

for seed in seed_prompts:
    result = generate_text(
        seed_text=seed,
        next_words=15,
        model=model,
        max_seq_len=max_sequence_len,
        temperature=0.75,
    )
    print(f"\n[Seed]: '{seed}'")
    print(f"[Generated]: {result}")