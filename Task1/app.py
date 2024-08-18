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
    qa_model = QuestionAnsweringModel("bert", "./outputs/best_model")
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


def predict_section_control(model, tokenizer, question, threshold=0.5, max_len=128):
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
    control_logits = logits[:, len(control_label_encoder.classes_):]

    section_probs = F.softmax(section_logits, dim=1)
    control_probs = F.softmax(control_logits, dim=1)

    section_confidence, section_pred = torch.max(section_probs, dim=1)
    control_confidence, control_pred = torch.max(control_probs, dim=1)

    if section_confidence.item() < threshold:
        section_label = "Unanswerable"
    else:
        section_label = reverse_section_label_mapping[section_pred.item()]

    if control_confidence.item() < threshold:
        control_label = "Unanswerable"
    else:
        control_label = reverse_control_label_mapping[control_pred.item()]

    return section_label, control_label

def process_question(question, model, tokenizer):
  section_pred, control_pred = predict_section_control(model, tokenizer, question)
  return section_pred, control_pred

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
    if answers[0]['answer'] == '' or answers[0]['answer'][0] == 'empty':
        return 'unanswerable'
    elif probabilities[0]['probability'][0] > 0.5:
        return 'answerable'
    else:
        return 'ambiguous'

st.title('Question Classification')

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
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

    q_df[['section_pred', 'control_pred']] = q_df['question'].apply(process_question, args=(section_model, tokenizer)).tolist()

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

    total_questions = len(classifications)
    answerable_questions = classifications.count('answerable')
    completion_percentage = (answerable_questions / total_questions) * 100

    st.write(f"Completion Percentage: {completion_percentage:.2f}%")
    st.write(f"\nTotal Questions: {total_questions}")
    st.write(f"Answerable Questions: {answerable_questions}")
    st.write(f"Unanswerable Questions: {classifications.count('unanswerable')}")
    st.write(f"Ambiguous Questions: {classifications.count('ambiguous')}")

    st.write("\nUnanswerable Questions:")
    for q in unanswerable_questions:
        st.write(f"- {q}")
