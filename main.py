# Modified code for Google Colab with Streamlit

from collections import Counter
import streamlit as st
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from collections import Counter
import io
import PyPDF2
import pandas as pd
import re  # Added import for regular expressions
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


nltk.download('punkt')


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text



def extract_skills(text):
    skills_keywords = ["python", "java", "machine learning",
                       "data analysis", "communication", "problem solving"]
    skills = [skill.lower()
              for skill in skills_keywords if skill.lower() in text.lower()]
    return skills


def preprocess_text(text):
    return word_tokenize(text.lower())

# Function to extract CGPA from text using regular expressions


def extract_cgpa(text):
    # Pattern for CGPA (e.g., 3.75, 4.0, etc.)
    cgpa_pattern = r'\b(\d\.\d{1,2})/\d{1,2}|\b(\d\.\d{1,2})\b'
    cgpa_matches = re.findall(cgpa_pattern, text)
    return [float(match[0]) if match[0] else 0 for match in cgpa_matches]


email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'


def train_doc2vec_model(documents):
    model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count,
                epochs=model.epochs)
    return model


def calculate_similarity(model, text1, text2):
    vector1 = model.infer_vector(preprocess_text(text1))
    vector2 = model.infer_vector(preprocess_text(text2))
    return model.dv.cosine_similarities(vector1, [vector2])[0]


def v_spacer(height, sb=False) -> None:
    for _ in range(height):
        if sb:
            st.sidebar.write('\n')
        else:
            st.write('\n')


# Streamlit Frontend
st.title("Resume Matching Tool📃📃")

# Sidebar - File Upload for Job Descriptions
st.sidebar.write("## Upload Job Description PDF")
job_descriptions_file = st.sidebar.file_uploader(
    "Upload Job Description PDF", type=["pdf"])

# Sidebar - File Upload for Resumes
st.sidebar.write("## Upload Resumes PDF")
resumes_files = st.sidebar.file_uploader(
    "Upload Resumes PDF", type=["pdf"], accept_multiple_files=True)

# Backend Processing
if job_descriptions_file is not None and resumes_files is not None:
    job_description_text = extract_text_from_pdf(job_descriptions_file)
    resumes_texts = [extract_text_from_pdf(resume_file) for resume_file in resumes_files]

    # Calculate skills for all resumes
    all_resumes_skills = [extract_skills(resume_text) for resume_text in resumes_texts]

    tagged_resumes = [TaggedDocument(words=preprocess_text(text), tags=[str(i)]) for i, text in enumerate(resumes_texts)]
    model_resumes = train_doc2vec_model(tagged_resumes)

    results_data = {'Resume': [], 'Similarity Score': [], 'CGPA': [], 'Total Score': [], 'Email': [], 'Contact': []}

    for i, resume_text in enumerate(resumes_texts):
        similarity_score = calculate_similarity(model_resumes, resume_text, job_description_text)
        cgpa_values = extract_cgpa(resume_text)
        cgpa = ', '.join(map(str, cgpa_values)) if cgpa_values else '0'
        results_data['Resume'].append(f"Resume {i+1}")
        smScore = similarity_score * 100
        total_score = smScore + sum(cgpa_values)
        results_data['Similarity Score'].append(smScore)
        results_data['CGPA'].append(cgpa)
        results_data['Total Score'].append(total_score)

        emails = ', '.join(re.findall(email_pattern, resume_text))
        contacts = ', '.join(re.findall(phone_pattern, resume_text))
        results_data['Email'].append(emails)
        results_data['Contact'].append(contacts)

    # Create a DataFrame
    results_df = pd.DataFrame(results_data)

    # Display the results table
    st.subheader("Results Table:")
    st.table(results_df)

    # Create a DataFrame for skills distribution
    skills_distribution_data = {'Resume': [], 'Skill': [], 'Frequency': []}
    for i, resume_skills in enumerate(all_resumes_skills):
        for skill in set(resume_skills):
            skills_distribution_data['Resume'].append(f"Resume {i+1}")
            skills_distribution_data['Skill'].append(skill)
            skills_distribution_data['Frequency'].append(resume_skills.count(skill))

    skills_distribution_df = pd.DataFrame(skills_distribution_data)

    # Pivot the DataFrame for heatmap
    skills_heatmap_df = skills_distribution_df.pivot(index='Resume', columns='Skill', values='Frequency').fillna(0)

    # Normalize the values for better visualization
    skills_heatmap_df_normalized = skills_heatmap_df.div(skills_heatmap_df.sum(axis=1), axis=0)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(skills_heatmap_df_normalized, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax)
    ax.set_title('Heatmap for Skills Distribution')
    ax.set_xlabel('Resume')
    ax.set_ylabel('Skill')

    # Display the Matplotlib figure using st.pyplot()
    st.pyplot(fig)
else:
    st.write("Upload the Job Description and Resumes PDF files to see the results.")
