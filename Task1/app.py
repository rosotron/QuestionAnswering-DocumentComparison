import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import json
import logging
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs

def load_models():
    section_model = BertForSequenceClassification.from_pretrained("./sectioncontrol/model")
    tokenizer = BertTokenizer.from_pretrained("./sectioncontrol/tokenizer")
    qa_model = QuestionAnsweringModel("bert", "/content/outputs/best_model")
    return section_model, tokenizer, qa_model

def preprocess(df):
    cols = list(df)[:2]
    df[cols] = df[cols].ffill(axis=0)
    df = df.iloc[4:]
    #df = df.dropna().reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df

section_model, tokenizer, qa_model = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
section_model.to(device)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, max_len):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]

        encoding = self.tokenizer.encode_plus(
            question,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Predict section and control headings
def predict_section_control(model, tokenizer, question, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        question,
        max_length=max_len,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    section_logits = logits[:, :len(section_label_encoder.classes_)]
    control_logits = logits[:, len(section_label_encoder.classes_):]

    section_probs = F.softmax(section_logits, dim=1)
    control_probs = F.softmax(control_logits, dim=1)

    section_confidence, section_pred = torch.max(section_probs, dim=1)
    control_confidence, control_pred = torch.max(control_probs, dim=1)

    return section_pred.item(), control_pred.item()

def get_context(section_heading, control_heading, df):
    filtered_df = df[(df['Section Heading'] == section_heading) | (df['Control Heading'] == control_heading)]
    notes = filtered_df['Notes/Comment'].dropna().tolist()

    a= ' '.join(notes) if notes else ''
    return a

def classify_question(question, section_heading, control_heading, df):
    context = get_context(section_heading, control_heading, df)
    if not context.strip():
        return 'unanswerable'
    to_predict = [
        {
            "context": context,
            "qas": [
                {
                    "question": question,
                    "id": "0",
                }
            ],
        }
    ]

    answers, probabilities = qa_model.predict(to_predict, n_best_size=2)
    if answers[0]['answer'] == '':
        return 'unanswerable'
    elif probabilities[0]['probability'][0] > 0.45:
        return 'answerable'
    else:
        return 'ambiguous'

# Streamlit App
st.title('Question Classification and Answering')

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    kb_df = pd.read_excel("/content/ClientABC _ ATB Financial_Knowledge Base.xlsx", "Data Sheet")
    q_df = pd.read_excel(uploaded_file, header=None)

    kb_df = preprocess(kb_df)
    q_df = q_df.rename(columns={0: 'question'})

    kb_df['Answer'] = kb_df['Answer'].fillna('unanswerable')

    section_label_encoder = LabelEncoder()
    control_label_encoder = LabelEncoder()

    kb_df['Section Heading Encoded'] = section_label_encoder.fit_transform(kb_df['Section Heading'])
    kb_df['Control Heading Encoded'] = control_label_encoder.fit_transform(kb_df['Control Heading'])

    reverse_section_label_mapping = dict(enumerate(section_label_encoder.classes_))
    reverse_control_label_mapping = dict(enumerate(control_label_encoder.classes_))

    # Predict section and control
    q_df[['section_pred', 'control_pred']] = q_df['question'].apply(
        lambda x: predict_section_control(section_model, tokenizer, x)).tolist()

    classifications = []
    unanswerable_questions = []

    for _, row in q_df.iterrows():
        question = row['question']
        section_heading = row['section_pred']
        control_heading = row['control_pred']

        classification = classify_question(question, section_heading, control_heading, kb_df)
        classifications.append(classification)

        if classification == 'unanswerable':
            unanswerable_questions.append(question)


    st.write(f"Completion Percentage: {(classifications.count('answerable') / len(classifications)) * 100:.2f}%")
    st.write(q_df)

    st.write("\nUnanswerable Questions:")
    for q in unanswerable_questions:
        st.write(f"- {q}")
