from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


# Frequently Asked Questions with alternative ways users may ask
faqs = [
    {
        "questions": [
            "What is your college name?",
            "What is the name of the college?",
            "Which college is this?",
            "Tell me your college name"
        ],
        "answer": "Our college is CodeAlpha Institute."
    },

    {
        "questions": [
            "What courses do you offer?",
            "Which courses are available?",
            "What programs do you have?",
            "What branches are available?"
        ],
        "answer": "We offer courses in Computer Science, Artificial Intelligence, Data Science, and related fields."
    },

    {
        "questions": [
            "What are the college timings?",
            "When does college start?",
            "When does college end?",
            "What time does college begin?",
            "What time does college finish?",
            "What are the college hours?"
        ],
        "answer": "College timings are from 9:00 AM to 4:00 PM."
    },

    {
        "questions": [
            "How can I contact the college?",
            "How do I contact the college?",
            "What is the college contact information?",
            "How can I reach the college administration?"
        ],
        "answer": "You can contact the college administration through the official college email or office."
    },

    {
        "questions": [
            "Where is the college located?",
            "What is the location of the college?",
            "Where is your college?",
            "What is the college address?"
        ],
        "answer": "The college is located in India."
    },

    {
        "questions": [
            "What is the admission process?",
            "How can I get admission?",
            "How do I apply for admission?",
            "What are the admission requirements?"
        ],
        "answer": "Students can apply through the official admission process and submit the required documents."
    },

    {
        "questions": [
            "Do you provide scholarships?",
            "Are scholarships available?",
            "Can students get scholarships?",
            "Is there any scholarship?"
        ],
        "answer": "Yes, scholarship opportunities may be available for eligible students."
    },

    {
        "questions": [
            "Is hostel facility available?",
            "Is there a hostel?",
            "Do you provide hostel facilities?",
            "Is accommodation available for students?"
        ],
        "answer": "Yes, hostel facilities are available for students."
    }
]


# Create a list containing all possible FAQ questions
all_questions = []
question_to_faq = []

for faq in faqs:
    for question in faq["questions"]:
        all_questions.append(question)
        question_to_faq.append(faq)


# Convert FAQ questions into numerical vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(all_questions)


def get_answer(user_question):

    # Convert user's question into a vector
    user_vector = vectorizer.transform([user_question])

    # Calculate similarity with all FAQ questions
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find the most similar question
    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    # Minimum similarity required
    if best_score < 0.20:
        return "Sorry, I couldn't find a relevant answer to your question."

    # Get the answer belonging to the matched FAQ
    best_faq = question_to_faq[best_match_index]

    return best_faq["answer"]


@app.route("/", methods=["GET", "POST"])
def home():

    answer = ""

    if request.method == "POST":

        user_question = request.form["question"]

        if user_question.strip():
            answer = get_answer(user_question)

    return render_template(
        "index.html",
        answer=answer
    )


if __name__ == "__main__":
    app.run(debug=True)