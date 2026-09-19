# AI FAQ Chatbot

An AI-powered FAQ chatbot developed as part of my CodeAlpha AI Internship.

## Features

- Answers frequently asked questions
- Understands differently worded questions
- Uses TF-IDF for text processing
- Uses cosine similarity to find the most relevant FAQ
- Simple and user-friendly chatbot interface
- Provides a fallback response for unknown questions
- Responsive web design

## Technologies Used

- Python
- Flask
- Scikit-learn
- HTML
- CSS
- TF-IDF
- Cosine Similarity

## How It Works

1. The user enters a question.
2. The question is converted into a numerical vector using TF-IDF.
3. The chatbot compares the question with stored FAQ questions.
4. Cosine similarity is used to find the most similar question.
5. The corresponding answer is displayed to the user.

## How to Run

### 1. Install the required packages

```bash
pip install flask scikit-learn nltk