import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.nn import Softmax
from torch import sort


@st.cache_resource  # кэширование
def load_model():
    return AutoModelForSequenceClassification.from_pretrained('zhdantim/mydeberta-v3-small').eval()


@st.cache_resource  # кэширование
def load_id2classes():
    return pd.read_csv('classes.tsv', sep='\t', index_col=0).to_dict()['classes']


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')


model = load_model()
id2classes = load_id2classes()
tokenizer = load_tokenizer()


def get_top_classes(text):
    tokenized_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    logits = model(**tokenized_text).logits.detach()
    probs = Softmax()(logits)
    probs_sorted, indices = sort(probs, descending=True)

    k = 1
    while sum(probs_sorted[0, :k]) < 0.95:
        k += 1

    return [id2classes[idx.item()] for idx in indices[0, :k]], probs_sorted[0, :k]

st.title("Простой классификатор статей")

title = st.text_input(label="Введите название статьи (обязательно)", value="Type Here ...")
abstract = st.text_input(label="Введите abstract", value="Type Here ...")
if st.button('Submit'):
    if title.title() != 'Type Here ...':
        if abstract.title() != 'Type Here ...':
            text = title.title() + '\n' + abstract.title()
        else:
            text = title.title()

        top_classes, probs = get_top_classes(text)

        for p, cls in zip(probs, top_classes):
            st.success(f'Статья относится к {cls} с вероятностью {p}')
    else:
        st.error('Введите название статьи')
