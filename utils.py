import torch 
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class customDataset(Dataset):
    def __init__(self, input_text, labels, tokenizer):
        self.input_text = input_text
        deceptive_labels, category_labels = zip(*labels)
        self.deceptive_labels = list(deceptive_labels)
        self.category_labels = list(category_labels)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenized_input_texts = self.tokenizer(
            self.input_text[idx],
            padding='max_length',
            truncation=True,
            max_length=128
        )
        return {
            'input_ids': torch.tensor(tokenized_input_texts['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_input_texts['attention_mask'], dtype=torch.long),
            'deceptive_labels': torch.tensor(self.deceptive_labels[idx], dtype=torch.long),
            'category_labels': torch.tensor(self.category_labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.input_text)


def preprocess_dataset(path, tokenizer, logger):
    df = pd.read_csv(path, sep=',')

    category_dict = {
        'Not Dark Pattern': 0,
        'Social Proof': 1,
        'Scarcity': 2,
        'Urgency': 3,
        'Misdirection': 4,
        'Sneaking': 5,
        'Obstruction': 6,
        'Forced Action': 7
    }

    input_text = df['text'].astype(str).tolist()
    deceptive_labels = df['label'].astype(int).tolist()
    category_labels = [category_dict[x] for x in df['Pattern Category']]
    labels = list(zip(deceptive_labels, category_labels))
    


    train_input_text, test_input_text, train_labels, test_labels = train_test_split(
        input_text, labels, test_size=0.2, random_state=42, stratify=deceptive_labels
    )

    deceptive_test_labels, _ = zip(*test_labels)
    deceptive_test_labels = list(deceptive_test_labels)

    validation_input_text, test_input_text, validation_labels, test_labels = train_test_split(
        test_input_text, test_labels, test_size=0.5, random_state=42, stratify=deceptive_test_labels
    )

    train_ds = customDataset(train_input_text, train_labels, tokenizer)
    validation_ds = customDataset(validation_input_text, validation_labels, tokenizer)
    test_ds = customDataset(test_input_text, test_labels, tokenizer)

    return train_ds, validation_ds, test_ds


def compute_metrics(metric, average, predictions, references):
    metric = evaluate.load(metric)

    if average:
        result = metric.compute(predictions=predictions, references=references, average=average)
    else:
        result = metric.compute(predictions=predictions, references=references)

    return result[metric.name]


def plot_metrics(x, y, title, xLabel, yLabel, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()