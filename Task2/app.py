

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

st.title('Version Comparison Tool')

def preprocess(df):
    cols = list(df)[:2]
    df[cols] = df[cols].ffill(axis=0)
    df = df.iloc[4:]
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df['Original ID'] = df['Original ID'].astype(str)
    return df

def create_wordcloud(text, title):
    text = ' '.join([str(t) for t in text if pd.notna(t)])
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        st.pyplot(plt)

def similarity_ratio(a, b):
    if pd.isna(a) and pd.isna(b):
        return 1.0
    elif pd.isna(a) or pd.isna(b):
        return 0.0

    a_enc = model.encode([str(a)])
    b_enc = model.encode([str(b)])

    similarity = cosine_similarity(a_enc, b_enc)[0][0]

    return similarity

def calculate_similarity_score(df1, df2, similarity_func):
    merged_df = df1.merge(df2, on='Original ID', suffixes=('_v1', '_final'))
    merged_df['Similarity_Score'] = merged_df.apply(lambda row: similarity_func(row['Answer_v1'], row['Answer_final']), axis=1)
    return merged_df[['Original ID', 'Answer_v1', 'Answer_final', 'Similarity_Score']]

def explain_similarity_histogram(similarity_scores):
    mean_score = np.mean(similarity_scores)
    median_score = np.median(similarity_scores)
    std_dev = np.std(similarity_scores)
    skewness = stats.skew(similarity_scores)
    kurtosis = stats.kurtosis(similarity_scores)

    explanation = f"""
    Explanation of the Similarity Scores Distribution:

       - The mean similarity score is {mean_score:.2f}.
       - The median similarity score is {median_score:.2f}.

       - The standard deviation of the scores is {std_dev:.2f}.

       - Skewness: {skewness:.2f} ({'positively' if skewness > 0 else 'negatively'} skewed)
         {' This suggests more scores are concentrated on the left side of the distribution.' if skewness > 0 else ' This suggests more scores are concentrated on the right side of the distribution.'}
       - Kurtosis: {kurtosis:.2f} ({'more' if kurtosis > 0 else 'less'} peaked than a normal distribution)
         {' This indicates more extreme values than a normal distribution.' if kurtosis > 0 else ' This indicates fewer extreme values than a normal distribution.'}

       - {'Most answers show high similarity between versions, suggesting minor changes overall.' if mean_score > 0.7 else 'Many answers show significant differences between versions, indicating more changes.'}
       - {'There is a wide range of similarity scores, indicating varied levels of changes across different questions.' if std_dev > 0.3 else 'The similarity scores are fairly consistent across questions, suggesting uniform levels of changes.'}
       - {'Some questions may have been completely rewritten or newly added, as indicated by very low similarity scores.' if np.min(similarity_scores) < 0.1 else 'The majority maintain similarity to the Version 1 document'}
    """
    st.write(explanation)

def categorize_similarity(score):
    if score < 0.3:
        return 'Completely Different'
    elif score < 0.7:
        return 'Somewhat Similar'
    else:
        return 'Very Similar'

def analyze_change_type(a, b):
    if pd.isna(a) and pd.isna(b):
        return 'Both Empty'
    elif pd.isna(a) and not pd.isna(b):
        return 'Added in Final'
    elif not pd.isna(a) and pd.isna(b):
        return 'Removed in Final'
    if str(a).strip().lower() == str(b).strip().lower():
        return 'No Change'
        
    a_words = set(word_tokenize(str(a).lower()))
    b_words = set(word_tokenize(str(b).lower()))
    common_words = a_words.intersection(b_words)

    if len(common_words) / max(len(a_words), len(b_words)) > 0.9:
        return 'Mainly Format/Grammar'
    else:
        return 'Content Change'

uploaded_file_v1 = st.file_uploader("Upload the first Excel file", type="xlsx")
uploaded_file_final = st.file_uploader("Upload the final Excel file", type="xlsx")

if uploaded_file_v1 is not None and uploaded_file_final is not None:
    df_v1 = pd.read_excel(uploaded_file_v1)
    df_final = pd.read_excel(uploaded_file_final)
    
    df_final = preprocess(df_final)
    
    cols = list(df_v1)[:2]
    df_v1[cols] = df_v1[cols].ffill(axis=0)
    df_v1['Original ID'] = df_v1['Original ID'].astype(str)

    st.subheader('Version 1')
    st.dataframe(df_v1.head())

    st.subheader('Final Version')
    st.dataframe(df_final.head())

    analysis_option = st.selectbox("Choose the analysis you want to perform:", 
                                   ["Word Clouds", 
                                    "Similarity Scores Distribution", 
                                    "Similarity Categories", 
                                    "Change Types", 
                                    "Summary Statistics", 
                                    "Questions with Significant Changes"])

    if analysis_option == "Word Clouds":
        st.subheader('Word Clouds')
        create_wordcloud(df_v1['Answer'], 'Word Cloud - Version 1 Answers')
        create_wordcloud(df_final['Answer'], 'Word Cloud - Final Version Answers')

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if analysis_option == "Similarity Scores Distribution":
        merged_df = calculate_similarity_score(df_v1, df_final, similarity_ratio)
        st.subheader('Similarity Scores Distribution')
        plt.figure(figsize=(10, 6))
        plt.hist(merged_df['Similarity_Score'], bins=20, edgecolor='black')
        plt.title('Distribution of Similarity Scores')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        explain_similarity_histogram(merged_df['Similarity_Score'])

    if analysis_option == "Similarity Categories":
        merged_df = calculate_similarity_score(df_v1, df_final, similarity_ratio)
        merged_df['Similarity_Category'] = merged_df['Similarity_Score'].apply(categorize_similarity)
        st.subheader('Simialrity Categories')
        plt.figure(figsize=(10, 6))
        merged_df['Similarity_Category'].value_counts().plot(kind='bar')
        plt.title('Distribution of Similarity Categories')
        plt.xlabel('Similarity Category')
        plt.ylabel('Count')
        st.pyplot(plt)

    if analysis_option == "Change Types":
        df_v1['Change_Type'] = df_v1.apply(
            lambda row: analyze_change_type(
                row['Answer'],
                df_final[df_final['Original ID'] == row['Original ID']]['Answer'].values[0]
            ) if not df_final[df_final['Original ID'] == row['Original ID']].empty else 'ID not found',
            axis=1
        )
        st.subheader('Change Types')
        plt.figure(figsize=(10, 6))
        df_v1['Change_Type'].value_counts().plot(kind='bar')
        plt.title('Distribution of Change Types')
        plt.xlabel('Change Type')
        plt.ylabel('Count')
        st.pyplot(plt)

    if analysis_option == "Summary Statistics":
        merged_df = calculate_similarity_score(df_v1, df_final, similarity_ratio)
        st.subheader('Summary Statistics')
        st.write(f"Total questions: {len(df_v1)}")
        st.write(f"Average similarity score: {merged_df['Similarity_Score'].mean():.2f}")
        
        st.subheader('Similarity Category Distribution')
        merged_df['Similarity_Category'] = merged_df['Similarity_Score'].apply(categorize_similarity)
        
        st.write(merged_df['Similarity_Category'].value_counts(normalize=True))
        
        df_v1['Change_Type'] = df_v1.apply(
            lambda row: analyze_change_type(
                row['Answer'],
                df_final[df_final['Original ID'] == row['Original ID']]['Answer'].values[0]
            ) if not df_final[df_final['Original ID'] == row['Original ID']].empty else 'ID not found',
            axis=1
        )
        st.subheader('Change Type Distribution')
        st.write(df_v1['Change_Type'].value_counts(normalize=True))

    if analysis_option == "Questions with Significant Changes":
        merged_df = calculate_similarity_score(df_v1, df_final, similarity_ratio)
        significant_changes = df_v1[merged_df['Similarity_Score'] < 0.5]
        st.subheader('Questions with Significant Changes')
        for idx, row in significant_changes.iterrows():
            st.write(f"\n**Question {idx + 1}:**")
            st.write(f"**Version 1:** {row['Answer']}")
            st.write(f"**Final Version:** {df_final.loc[idx, 'Answer']}")
