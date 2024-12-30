################################################
#### esse exemplo fica pra depois.
#### compatibilizar código atual com a codebase
################################################


# import torch
# from torch import nn

# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator, GloVe
# from torchtext.transforms import Sequential, ToTensor, VocabTransform

# import warnings

# warnings.filterwarnings("ignore")


# tokenizer = get_tokenizer("spacy", language="en_core_web_sm")


# def yield_tokens(data_iter):
#     for text in data_iter:
#         yield tokenizer(text)


# def collate_batch(batch):
#     pass


# def _preprocessing(data_iter):
#     vocab = yield_tokens(data_iter)
#     vocab = build_vocab_from_iterator(vocab, specials=["<unk>"])
#     vocab.set_default_index(vocab["<unk>"])
#     vocab = VocabTransform(vocab)
#     return Sequential(vocab, ToTensor())


# def collate_batch(batch):
#     label, text = [], []
#     for lbl, txt in batch:
#         label.append(lbl)
#         text.append(txt)


# def preprocessing(data):
#     data_ = _preprocessing(data)


# class NLPClassifier(nn.Module):
#     pass


# if __name__ == "__main__":
#     import pickle

#     DATA_HOME = "/Users/victor/Documents/projects/coreset/data/drugs_review/transformed_drugs_review.pickle"
#     with open(DATA_HOME, "rb") as file:
#         features = pickle.load(file)
#     features, target = features["features"], features["target"]
#     features = _preprocessing(features)

# print(features)

import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torchtext.transforms import Sequential, ToTensor, VocabTransform

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from torch_utils.data import sampling_dataset, BaseDataset

import pickle

DATA_HOME = "/Users/victor/Documents/projects/coreset/data/drugs_review/transformed_drugs_review.pickle"
with open(DATA_HOME, "rb") as file:
    features = pickle.load(file)

# 1. Pré-processamento do texto
tokenizer = get_tokenizer("spacy", language="en_core_web_sm")


def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)
    # yield from map(tokenizer, data_iter)


def clean_sent(sent, sub_pattern=r"[\W\s]+"):
    # sent = " ".join(sent).lower()
    sent = sent.lower()
    sent = re.sub(sub_pattern, " ", sent)
    sent = re.split(r"\W", sent)
    sent = " ".join(filter(lambda x: x.isalnum() and not x.isdigit(), sent))
    return sent


features, y_train = features["features"], features["target"]
y_train = list(map(lambda x: 1 if x > 5 else 0, y_train))
X_train = list(map(clean_sent, features))

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

vocab = build_vocab_from_iterator(
    yield_tokens(X_train), specials=["<unk>"], max_tokens=10000
)
vocab.set_default_index(vocab["<unk>"])

# Transforma o texto para tensores de índices de vocabulário
text_transform = Sequential(VocabTransform(vocab), ToTensor())


# Função para converter o texto e rótulo em tensores
def collate_batch(batch):
    label_list, text_list = [], []
    # for _label, _text in batch:
    for _label, _text in batch:
        label_list.append(_label)  # Ajustando para começar de 0
        processed_text = torch.tensor(
            text_transform(tokenizer(_text)), dtype=torch.long
        )
        text_list.append(processed_text)
    # Preenchendo as sequências para que tenham o mesmo comprimento no batch
    text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    return text_list, label_list


# 2. Criando o DataLoader para treinamento
batch_size = 1024
# train_iter = BaseDataset(list(train_iter), list(target))
train_dataloader = DataLoader(
    list(zip(y_train, X_train)),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)

X_test, y_test = collate_batch(list(zip(y_test, X_test)))


# 3. Definindo o modelo de classificação
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.out_layer = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        pooled = embedded.mean(dim=1)
        out_ = self.fc(pooled)

        return self.out_layer(out_)


# Inicializando o modelo, função de perda e otimizador
vocab_size = len(vocab)
embedding_dim = 100
output_dim = 1

model = SimpleNN(vocab_size, embedding_dim, output_dim)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()


# 4. Função de treino
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for text, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# 5. Treinando o modelo
n_epochs = 15
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    print(f"Época {epoch+1}: Perda de Treino = {train_loss:.3f}")

pred = model(X_test).detach().numpy().round()
print(classification_report(y_pred=pred, y_true=y_test))
