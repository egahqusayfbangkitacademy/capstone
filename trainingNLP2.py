import json
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Membaca dataset dari file JSONL dengan penanganan error
def load_dataset(file_path):
    questions = []
    answers = []
    categories = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    questions.append(data['pertanyaan'])
                    answers.append(data['jawaban'][0] if data['jawaban'] else 'tidak_diketahui')
                    categories.append(data['kategori'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
    except FileNotFoundError:
        print(f"File tidak ditemukan: {file_path}")
        questions = ["Apa makanan sapi?", "Berapa kaki kucing?"]
        answers = ["rumput", "empat"]
        categories = ["sapi", "kucing"]
    
    return questions, answers, categories

# Fungsi untuk membuat model
def create_model(vocab_size_questions, vocab_size_answers, category_size, max_question_length):
    input_questions = tf.keras.Input(shape=(max_question_length,))
    input_embeddings = tf.keras.layers.Embedding(vocab_size_questions, 64)(input_questions)
    lstm_layer = tf.keras.layers.LSTM(128)(input_embeddings)
    
    dense_answers = tf.keras.layers.Dense(vocab_size_answers, activation='softmax', name='answer_output')(lstm_layer)
    dense_categories = tf.keras.layers.Dense(category_size, activation='softmax', name='category_output')(lstm_layer)
    
    model = tf.keras.Model(inputs=input_questions, outputs=[dense_answers, dense_categories])
    model.compile(
        optimizer='adam',
        loss={
            'answer_output': 'sparse_categorical_crossentropy', 
            'category_output': 'categorical_crossentropy'
        },
        metrics={
            'answer_output': 'accuracy', 
            'category_output': 'accuracy'
        }
    )
    return model

def save_tokenizers_config(save_path, tokenizer_questions, tokenizer_answers, tokenizer_categories, max_question_length, max_answer_length):
    """Menyimpan konfigurasi tokenizer dalam format JSON"""
    os.makedirs(save_path, exist_ok=True)
    
    tokenizers_config = {
        'questions': {
            'word_index': tokenizer_questions.word_index,
            'index_word': tokenizer_questions.index_word,
            'word_counts': tokenizer_questions.word_counts,
            'document_count': tokenizer_questions.document_count,
        },
        'answers': {
            'word_index': tokenizer_answers.word_index,
            'index_word': tokenizer_answers.index_word,
            'word_counts': tokenizer_answers.word_counts,
            'document_count': tokenizer_answers.document_count,
        },
        'categories': {
            'word_index': tokenizer_categories.word_index,
            'index_word': tokenizer_categories.index_word,
            'word_counts': tokenizer_categories.word_counts,
            'document_count': tokenizer_categories.document_count,
        },
        'max_lengths': {
            'question': max_question_length,
            'answer': max_answer_length
        }
    }
    
    with open(f'{save_path}/tokenizers.json', 'w', encoding='utf-8') as f:
        json.dump(tokenizers_config, f, ensure_ascii=False, indent=2)

def save_model_weights(model, save_path):
    """Menyimpan weights model dalam format binary"""
    weights = model.get_weights()
    
    # Menyimpan metadata weights
    metadata = {
        'layer_names': [layer.name for layer in model.layers],
        'weight_shapes': [list(w.shape) for w in weights],
        'weight_dtypes': [str(w.dtype) for w in weights]
    }
    
    with open(f'{save_path}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Menyimpan weights dalam format binary
    with open(f'{save_path}/weights.bin', 'wb') as f:
        for weight in weights:
            f.write(weight.tobytes())

def load_saved_model(load_path):
    """Memuat model dan tokenizer dari file yang disimpan"""
    # Memuat tokenizer
    with open(f'{load_path}/tokenizers.json', 'r', encoding='utf-8') as f:
        tokenizers_config = json.load(f)
    
    # Membuat dan mengonfigurasi tokenizer
    tokenizer_questions = Tokenizer()
    tokenizer_answers = Tokenizer()
    tokenizer_categories = Tokenizer()
    
    tokenizer_questions.word_index = tokenizers_config['questions']['word_index']
    tokenizer_answers.word_index = tokenizers_config['answers']['word_index']
    tokenizer_categories.word_index = tokenizers_config['categories']['word_index']
    
    # Memuat metadata
    with open(f'{load_path}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Memuat weights
    weights = []
    with open(f'{load_path}/weights.bin', 'rb') as f:
        for shape, dtype in zip(metadata['weight_shapes'], metadata['weight_dtypes']):
            size = np.prod(shape)
            weight_bytes = f.read(size * np.dtype(dtype).itemsize)
            weight = np.frombuffer(weight_bytes, dtype=dtype).reshape(shape)
            weights.append(weight)
    
    # Membuat model baru
    model = create_model(
        vocab_size_questions=len(tokenizer_questions.word_index) + 1,
        vocab_size_answers=len(tokenizer_answers.word_index) + 1,
        category_size=len(tokenizer_categories.word_index) + 1,
        max_question_length=tokenizers_config['max_lengths']['question']
    )
    
    # Set weights ke model
    model.set_weights(weights)
    
    return model, tokenizer_questions, tokenizer_answers, tokenizer_categories

def predict_answer(question, model, tokenizer_q, tokenizer_a, tokenizer_c, max_len):
    """Fungsi untuk memprediksi jawaban"""
    # Tokenisasi pertanyaan input
    question_seq = tokenizer_q.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen=max_len, padding='post')
    
    # Prediksi
    answer_pred, category_pred = model.predict(question_padded)
    
    # Decode prediksi
    predicted_answer_index = np.argmax(answer_pred)
    predicted_category_index = np.argmax(category_pred)
    
    # Konversi ke teks
    predicted_answer = list(tokenizer_a.word_index.keys())[list(tokenizer_a.word_index.values()).index(predicted_answer_index)] if predicted_answer_index in tokenizer_a.word_index.values() else "Tidak diketahui"
    predicted_category = list(tokenizer_c.word_index.keys())[list(tokenizer_c.word_index.values()).index(predicted_category_index)] if predicted_category_index in tokenizer_c.word_index.values() else "Tidak diketahui"
    
    return predicted_answer, predicted_category

def main():
    # Path file dataset 
    dataset_path = 'dataset.jsonl'
    questions, answers, categories = load_dataset(dataset_path)

    # Tokenisasi pertanyaan
    tokenizer_questions = Tokenizer()
    tokenizer_questions.fit_on_texts(questions)
    question_sequences = tokenizer_questions.texts_to_sequences(questions)
    max_question_length = max(len(seq) for seq in question_sequences)
    question_padded = pad_sequences(question_sequences, maxlen=max_question_length, padding='post')

    # Tokenisasi jawaban
    tokenizer_answers = Tokenizer()
    tokenizer_answers.fit_on_texts(answers)
    answer_sequences = tokenizer_answers.texts_to_sequences(answers)
    max_answer_length = max(len(seq) for seq in answer_sequences) if answer_sequences else 1
    answer_padded = pad_sequences(answer_sequences, maxlen=max_answer_length, padding='post')

    # Tokenisasi kategori
    tokenizer_categories = Tokenizer()
    tokenizer_categories.fit_on_texts(categories)
    category_sequences = tokenizer_categories.texts_to_sequences(categories)
    categories_encoded = to_categorical(category_sequences)

    # Split data
    X_train, X_test, y_answer_train, y_answer_test, y_category_train, y_category_test = train_test_split(
        question_padded, answer_padded, categories_encoded, test_size=0.2, random_state=42
    )

    # Buat dan compile model
    vocab_size_questions = len(tokenizer_questions.word_index) + 1
    vocab_size_answers = len(tokenizer_answers.word_index) + 1
    category_size = len(tokenizer_categories.word_index) + 1

    model = create_model(
        vocab_size_questions=vocab_size_questions,
        vocab_size_answers=vocab_size_answers,
        category_size=category_size,
        max_question_length=max_question_length
    )

    # Training
    history = model.fit(
        X_train, 
        {
            'answer_output': np.argmax(y_answer_train, axis=-1), 
            'category_output': y_category_train
        },
        epochs=1000,
        validation_split=0.2,
        verbose=1
    )

    # Evaluasi
    model.evaluate(X_test, {
        'answer_output': np.argmax(y_answer_test, axis=-1), 
        'category_output': y_category_test
    })

    # Simpan model
    save_path = './saved_model'
    os.makedirs(save_path, exist_ok=True)
    save_tokenizers_config(save_path, tokenizer_questions, tokenizer_answers, tokenizer_categories, 
                          max_question_length, max_answer_length)
    save_model_weights(model, save_path)

    # Contoh prediksi
    contoh_pertanyaan = "Sapi makan apa setiap hari?"
    jawaban, kategori = predict_answer(
        contoh_pertanyaan,
        model=model,
        tokenizer_q=tokenizer_questions,
        tokenizer_a=tokenizer_answers,
        tokenizer_c=tokenizer_categories,
        max_len=max_question_length
    )
    print("\nMenggunakan model yang baru ditraining:")
    print(f"Pertanyaan: {contoh_pertanyaan}")
    print(f"Jawaban Prediksi: {jawaban}")
    print(f"Kategori: {kategori}")

    # Test dengan model yang dimuat
    loaded_model, loaded_tokenizer_q, loaded_tokenizer_a, loaded_tokenizer_c = load_saved_model('./saved_model')
    jawaban, kategori = predict_answer(
        contoh_pertanyaan,
        model=loaded_model,
        tokenizer_q=loaded_tokenizer_q,
        tokenizer_a=loaded_tokenizer_a,
        tokenizer_c=loaded_tokenizer_c,
        max_len=max_question_length
    )
    print("\nMenggunakan model yang dimuat:")
    print(f"Pertanyaan: {contoh_pertanyaan}")
    print(f"Jawaban Prediksi: {jawaban}")
    print(f"Kategori: {kategori}")

if __name__ == "__main__":
    main()