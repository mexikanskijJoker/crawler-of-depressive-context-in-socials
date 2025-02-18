import numpy as np
import pandas as pd
import torch
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, classification_report

tqdm.pandas()

# 2. Загрузка данных
df = pd.read_excel("data/dataset.xlsx")

depression = ['depression', 'no depression']

# 3. Разделение данных на обучающую и валидационную выборки
train_df, val_df = train_test_split(df, test_size=0.2, stratify = df['label'], random_state = 42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_name = "DeepPavlov/rubert-base-cased"
depression_base = AutoModel.from_pretrained(model_name).to(device)
depression_classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
depression_tokenizer = AutoTokenizer.from_pretrained(model_name)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embeds(text, model, tokenizer):
    # Ensure the text is a string
    if not isinstance(text, str):
        text = str(text)
    
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings[0].to(device)

# Ensure all text values are strings and handle missing values
train_df['text'] = train_df['text'].fillna('').astype(str)
val_df['text'] = val_df['text'].fillna('').astype(str)

train_df['embedding'] = train_df['text'].progress_apply(get_embeds, model=depression_base, tokenizer=depression_tokenizer)
val_df['embedding'] = val_df['text'].progress_apply(get_embeds, model=depression_base, tokenizer=depression_tokenizer)

train_encodings = depression_tokenizer(train_df['text'].tolist(),
                                    truncation=True, padding=True,
                                    max_length = 374)
val_encodings = depression_tokenizer(val_df['text'].tolist(),
                                  truncation=True, padding=True,
                                  max_length = 374)


class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = DepressionDataset(encodings=train_encodings,
                              labels=train_df['label'].values)
valid_dataset = DepressionDataset(encodings=val_encodings,
                              labels=val_df['label'].values)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    f1 = f1_score(labels, preds, average='macro')
    f1_micro = f1_score(labels, preds, average='micro')
    return {
            'f1 macro'      : f1,
            'f1 micro'      : f1_micro,
            }

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=5e-5,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=2500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=2500,
    save_steps=2500,
    fp16=True,
    report_to="none",
    evaluation_strategy="steps")

trainer = Trainer(
    model=depression_classifier.to(device),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

test_pred = trainer.predict(valid_dataset)

# Метрики качества по классам, минимальные и средние метрики, Accuracy по топ-1и топ-2, матрица ошибок

def quality(y_predict, y_true):
    # Convert predictions to class labels
    y_pred = np.argmax(y_predict, axis=1)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_avg = precision_score(y_true, y_pred, average='weighted')
    recall_avg = recall_score(y_true, y_pred, average='weighted')
    fscore_avg = f1_score(y_true, y_pred, average='weighted')

    # Metrics for each class
    report = precision_recall_fscore_support(y_true, y_pred)

    precision_min = report[0].min()
    recall_min = report[1].min()
    fscore_min = report[2].min()

    precision_dict = {'min': precision_min, 'avg': precision_avg}
    precision = pd.Series(precision_dict)
    recall_dict = {'min': recall_min, 'avg': recall_avg}
    recall = pd.Series(recall_dict)
    fscore_dict = {'min': fscore_min, 'avg': fscore_avg}
    fscore = pd.Series(fscore_dict)
    quality_df = pd.DataFrame({'precision': precision, 'recall': recall, 'f1-score': fscore})

    # All metrics
    print(classification_report(y_true, y_pred, target_names=depression))
    print()
    print(f'Accuracy: {accuracy:.4f}')
    print()
    print(quality_df)
    print()

# Use the updated quality function
quality(y_predict=test_pred.predictions, y_true=val_df['label'])


# 8. Сохранение модели и токенизатора
depression_classifier.save_pretrained("./depression_classifier")
depression_tokenizer.save_pretrained("./depression_classifier")

# Перемещаем модель на выбранное устройство
depression_classifier.to(device)

# Токенизируем и подготавливаем входные тензоры
test_texts = [
    "Я уже несколько недель не могу найти в себе силы даже встать с кровати и выйти на улицу.",
    "Все вокруг кажется серым, я не вижу смысла в будущем и в жизни в целом.",
    "Сегодня я выполнил все задачи, которые планировал, и чувствую себя довольным!",
    "Мне тревожно, я постоянно переживаю из-за мелочей и не могу успокоиться.",
    "Я ощущаю внутренний подъем, у меня есть силы и желание заниматься любимыми делами!",
    "Деградация человечества неумолима. Лучшие мечтают о том, чтобы навсегда уйти.",
    "Можно сказать, что живу я в достатке. Можно сказать, что живу я в достатке."
]
inputs = depression_tokenizer(test_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Перемещаем входные тензоры на то же устройство, что и модель
inputs = {key: value.to(device) for key, value in inputs.items()}

# Делаем предсказания
with torch.no_grad():
    outputs = depression_classifier(**inputs)

# Конвертируем предсказания в метки классов
predictions = torch.argmax(outputs.logits, dim=-1).cpu()  # Перемещаем предсказания обратно на CPU, если необходимо

# Выводим результаты
for text, label in zip(test_texts, predictions.numpy()):
    print(f"Текст: {text} | Предсказание: {'Депрессия' if label == 1 else 'Нет депрессии'}")
