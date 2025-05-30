from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys 

os.environ["WANDB_DISABLED"] = "true" 

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s:;)(]', ' ', text)
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    text = re.sub(r':D', ' very happy ', text)
    return text.strip()

def load_and_preprocess_data(filepath, sample_fraction=1.0): 
    try:
        df = pd.read_csv(filepath)
        if 'text' not in df.columns or 'label' not in df.columns:
            if len(df.columns) == 2:
                print(f"Nie znaleziono kolumn 'text' i 'label'. Używam pierwszej ({df.columns[0]}) i drugiej ({df.columns[1]}).")
                df.columns = ['text', 'label']
            else:
                print(f"Nie znaleziono kolumn 'text' i 'label'. Kolumny: {df.columns.tolist()}. Zakładam pierwszą jako tekst, ostatnią jako etykietę.")
                df = df.rename(columns={df.columns[0]: 'text', df.columns[-1]: 'label'})

        if sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)

        df['processed_text'] = df['text'].apply(preprocess_text)
        df = df[df['processed_text'].str.len() > 0].reset_index(drop=True)

        print(f"Wczytano {len(df)} próbek po preprocessingu.")
        print("Rozkład klas:")
        print(df['label'].value_counts(dropna=False))
        return df
    except Exception as e:
        print(f"Błąd wczytywania danych (standardowo): {e}")
        try:
            print("Próba wczytania danych z ręcznym parsowaniem...")
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f.readlines() if line.strip()]
            header_line = all_lines[0]
            data_lines = all_lines[1:]
            parsed_data = []
            for line_content in data_lines:
                parts = line_content.rsplit(',', 1)
                if len(parts) == 2:
                    text_val, label_val = parts[0].strip('" '), parts[1].strip('" ')
                    if text_val and label_val:
                        parsed_data.append({'text': text_val, 'label': label_val})
            df_manual = pd.DataFrame(parsed_data)
            if sample_fraction < 1.0:
                df_manual = df_manual.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
            df_manual['processed_text'] = df_manual['text'].apply(preprocess_text)
            df_manual = df_manual[df_manual['processed_text'].str.len() > 0].reset_index(drop=True)
            print(f"Wczytano (ręcznie) {len(df_manual)} próbek po preprocessingu.")
            print("Rozkład klas (ręcznie):")
            print(df_manual['label'].value_counts(dropna=False))
            return df_manual
        except Exception as e_manual:
            print(f"Błąd wczytywania danych (również przy ręcznym parsowaniu): {e_manual}")
            return None

def create_balanced_dataset(df, min_samples_per_class=10): 
    class_counts = df['label'].value_counts()
    print(f"Oryginalne rozkłady klas:\n{class_counts}")

    valid_classes = class_counts[class_counts >= min_samples_per_class].index
    df_filtered = df[df['label'].isin(valid_classes)].copy()

    if df_filtered.empty or df_filtered['label'].nunique() == 0: 
        print("Brak wystarczającej liczby próbek w klasach do zbalansowania lub brak klas. Zwracam oryginalny DataFrame.")
        return df

    min_count_possible = df_filtered['label'].value_counts().min()
    
    if not valid_classes.empty:
        target_count = min(min_count_possible, class_counts[valid_classes].min())
    else: 
        target_count = min_count_possible


    balanced_dfs = []
    for label_val in df_filtered['label'].unique():
        class_df = df_filtered[df_filtered['label'] == label_val]
        n_samples = min(len(class_df), target_count)
        if n_samples > 0:
            sampled_df = class_df.sample(n=n_samples, random_state=42)
            balanced_dfs.append(sampled_df)
    
    if not balanced_dfs:
        print("Nie udało się zbalansować datasetu. Zwracam DataFrame po filtrowaniu (jeśli było).")
        return df_filtered 

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Zbalansowany dataset: {len(balanced_df)} próbek.")
    print("Nowe rozkłady klas:")
    print(balanced_df['label'].value_counts())
    return balanced_df

def train_emotion_classifier(df, model_name="distilbert-base-uncased", test_size=0.2):
    if df.empty or 'label' not in df.columns or 'processed_text' not in df.columns:
        print("Brak danych do trenowania lub brak wymaganych kolumn.")
        return None, None, None

    label_encoder = LabelEncoder()
    df['encoded_labels'] = label_encoder.fit_transform(df['label'])

    if df['encoded_labels'].nunique() < 2:
        print("Potrzebne co najmniej 2 klasy do stratyfikacji. Używam podziału bez stratyfikacji.")
        stratify_labels = None
    else:
        min_class_count_for_split = df['encoded_labels'].value_counts().min()
        if min_class_count_for_split < 2 and test_size > 0:
            print(f"Jedna z klas ma mniej niż 2 próbki ({min_class_count_for_split}). Używam podziału bez stratyfikacji.")
            stratify_labels = None
        else:
            stratify_labels = df['encoded_labels']


    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'].tolist(),
        df['encoded_labels'].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        ignore_mismatched_sizes=True
    )

    train_dataset = EmotionDataset(X_train, y_train, tokenizer)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer)

    training_args = TrainingArguments(
        output_dir='./emotion_classifier_output',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer, 
    )

    print("Rozpoczynanie treningu...")
    trainer.train()

    print("Ewaluacja modelu...")
    predictions_output = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions_output.predictions, axis=1)

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    print(f"\nDokładność po fine-tuningu: {accuracy:.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, cmap='Blues')
    plt.title('Macierz pomyłek')
    plt.ylabel('Rzeczywiste etykiety')
    plt.xlabel('Przewidziane etykiety')
    plt.show()

    return model, tokenizer, label_encoder

def main():
    filepath = "C:/Users/20pat/OneDrive/Pulpit/Szkoła/UM_DSP/emotionsDataset.csv"

    df_full = load_and_preprocess_data(filepath, sample_fraction=1.0) 

    if df_full is not None and not df_full.empty:
        print("\n--- Przygotowywanie modelu Fine-tuned ---")
        balanced_df = create_balanced_dataset(df_full, min_samples_per_class=10)
        
        if balanced_df.empty or len(balanced_df) < 2 * len(balanced_df['label'].unique()):
            print("Za mało danych po balansowaniu do przeprowadzenia treningu i testu.")
            model, tokenizer, label_encoder = None, None, None
        else:
            model, tokenizer, label_encoder = train_emotion_classifier(balanced_df)

        if model and tokenizer and label_encoder:
            print("\n=== TESTOWANIE NA NOWYCH PRZYKŁADACH (model fine-tuned) ===")
            test_texts = [
                "I am so happy today!",
                "This makes me really angry",
                "I feel very sad about this situation",
                "What a surprising turn of events!",
                "I love spending time with my family"
            ]

            device = 0 if torch.cuda.is_available() else -1
            print(f"Pipeline text-classification będzie używał urządzenia: {'GPU' if device == 0 else 'CPU'}")

            classifier_ft = pipeline("text-classification",
                                     model=model,
                                     tokenizer=tokenizer,
                                     device=device)

            processed_test_texts = [preprocess_text(text) for text in test_texts]
            all_results_ft = classifier_ft(processed_test_texts)

            for i, original_text in enumerate(test_texts):
                result = all_results_ft[i]
                if 'label' in result and isinstance(result['label'], str) and '_' in result['label']:
                    predicted_id_str = result['label'].split('_')[-1]
                    if predicted_id_str.isdigit():
                        predicted_id = int(predicted_id_str)
                        if predicted_id < len(label_encoder.classes_):
                             predicted_emotion = label_encoder.inverse_transform([predicted_id])[0]
                             confidence = result['score']
                             print(f"Tekst: '{original_text}'")
                             print(f"  Przewidziana emocja: {predicted_emotion} (pewność: {confidence:.4f})")
                        else:
                            print(f"Tekst: '{original_text}' - Błąd: Przewidziany ID {predicted_id} poza zakresem klas ({len(label_encoder.classes_)}).")
                    else:
                        print(f"Tekst: '{original_text}' - Błąd: Nie można sparsować ID z etykiety {result['label']}.")
                else:
                    print(f"Tekst: '{original_text}' - Błąd: Nieoczekiwany format wyniku: {result}")
                print()
        else:
            print("Nie udało się wytrenować modelu fine-tuned. Testowanie na nowych przykładach pominięte.")
    else:
        print("Nie udało się wczytać danych do analizy.")

if __name__ == "__main__":
    main()
