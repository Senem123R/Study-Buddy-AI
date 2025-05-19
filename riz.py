import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
from docx import Document
import os
import json
import warnings
from PyPDF2.errors import PdfReadError
from google.api_core import exceptions as google_exceptions
from langchain.globals import set_verbose

# Suppress langchain verbose import warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Importing verbose from langchain root module is no longer supported."
)
set_verbose(False)

# Validate Gemini API key
def validate_api_key():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        llm.invoke("Test")
        return True
    except google_exceptions.InvalidArgument as e:
        st.error(f"Invalid Gemini API key: {str(e)}. Please provide a valid key.")
        return False
    except Exception as e:
        st.error(f"Error validating API key: {str(e)}. Please check your key and try again.")
        return False

# Set up Gemini API key
if "google_api_key" not in st.session_state:
    st.session_state.google_api_key = os.environ.get("GOOGLE_API_KEY", "")
if not st.session_state.google_api_key:
    st.session_state.google_api_key = st.text_input("Enter your Gemini API key:", type="password")
    if not st.session_state.google_api_key:
        st.error("AIzaSyCrgOdqrwLXmgbnP8PZ27ZVQrvm7OMkCk8 is not a valid API key. Please enter a valid key.")
        st.stop()
os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key

# Validate API key
if not validate_api_key():
    st.stop()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_output_tokens=4000
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# Prompt templates
study_plan_template = PromptTemplate(
    input_variables=["topic", "time", "chat_history"],
    template="""
    You are a helpful study assistant. Based on the user's topic "{topic}" and available time "{time}" hours,
    create a concise study plan. Use the conversation history to personalize the plan: {chat_history}.
    Provide a step-by-step plan with specific tasks and time allocations.
    """
)
answer_template = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
    You are a knowledgeable study assistant. Answer the following question clearly and concisely: "{question}".
    Use the conversation history to provide context if relevant: {chat_history}.
    """
)
file_answer_template = PromptTemplate(
    input_variables=["question", "file_content", "chat_history"],
    template="""
    You are a study assistant. Answer the following question based on the provided lecture notes: "{question}".
    Lecture notes: {file_content}
    Use the conversation history for additional context if relevant: {chat_history}.
    If the lecture notes don't contain relevant information, say so and provide a general answer if possible.
    """
)
file_quiz_template = PromptTemplate(
    input_variables=["file_content", "chat_history"],
    template="""
    You are a study assistant creating a quiz. Based on the provided lecture notes, generate a multiple-choice question (with 4 options and 1 correct answer) that tests understanding of a key concept or fact in the notes: {file_content}.
    Use the conversation history for context: {chat_history}.
    Ensure the question is directly related to the content of the notes and avoid general knowledge questions.
    Format the output exactly as:
    Question: [Your question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Correct option]
    """
)
quiz_template = PromptTemplate(
    input_variables=["topic", "chat_history"],
    template="""
    You are a study assistant creating a quiz. Generate a multiple-choice question (with 4 options and 1 correct answer)
    based on the topic "{topic}". Use the conversation history for context: {chat_history}.
    Format the output as:
    Question: [Your question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Correct option]
    """
)

# Initialize LangChain chains
def create_chain(prompt):
    return RunnableSequence(prompt | llm)

study_plan_chain = RunnableWithMessageHistory(
    create_chain(study_plan_template),
    lambda session_id: st.session_state.chat_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history"
)
answer_chain = RunnableWithMessageHistory(
    create_chain(answer_template),
    lambda session_id: st.session_state.chat_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history"
)
file_answer_chain = RunnableWithMessageHistory(
    create_chain(file_answer_template),
    lambda session_id: st.session_state.chat_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history"
)
file_quiz_chain = RunnableWithMessageHistory(
    create_chain(file_quiz_template),
    lambda session_id: st.session_state.chat_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history"
)
quiz_chain = RunnableWithMessageHistory(
    create_chain(quiz_template),
    lambda session_id: st.session_state.chat_history,
    input_messages_key="chat_history",
    history_messages_key="chat_history"
)

# Format chat history
def format_chat_history(chat_history):
    formatted = ""
    for msg in chat_history.messages:
        role = "User" if msg.type == "human" else "Assistant"
        formatted += f"{role}: {msg.content}\n"
    return formatted

# Extract text from files
def extract_file_text(uploaded_file):
    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
            return text if text else "No text could be extracted from the PDF."
        elif file_extension == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
            return text if text else "No text could be extracted from the Word document."
        elif file_extension == "txt":
            text = uploaded_file.read().decode("utf-8")
            return text if text else "No text could be extracted from the text file."
        else:
            return "Unsupported file format. Please upload a PDF, Word, or text file."
    except PdfReadError:
        return "Error: The PDF file is corrupted or invalid."
    except Exception as e:
        return f"Error extracting file text: {str(e)}"

# Save progress to JSON
def save_progress():
    with open("progress.json", "w") as f:
        json.dump(st.session_state.progress, f)

# Load progress from JSON
def load_progress():
    try:
        with open("progress.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "quiz_results": [],
            "study_plans": 0,
            "quizzes_attempted": 0,
            "quizzes_correct": 0
        }

# Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Study Buddy AI", initial_sidebar_state="expanded")
    # Apply 9:16 aspect ratio CSS for mobile
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 720px; /* Base width for 9:16 */
            aspect-ratio: 9 / 16;
            height: 1280px; /* Example height for 9:16 */
            margin: auto; /* Center the container */
            padding: 20px;
            overflow: auto; /* Handle overflow content */
            box-sizing: border-box;
        }
        /* Ensure content is centered and avoids edges (e.g., for mobile apps) */
        .main .block-container > * {
            padding-top: 250px; /* Avoid top/bottom UI overlays */
            padding-bottom: 250px;
        }
        /* Responsive adjustments for smaller screens */
        @media (max-width: 720px) {
            .main .block-container {
                max-width: 100%;
                height: calc(100vw * 16 / 9); /* Maintain 9:16 ratio */
            }
        }
        /* Force portrait orientation */
        @media (orientation: landscape) {
            html::after {
                content: "Please rotate your device to portrait mode.";
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                background: #000;
                color: #fff;
                font-size: 24px;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: 9999;
            }
            .main {
                display: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Study Buddy AI")
    st.write("Your personal assistant for study plans, answering questions, file-based Q&A, and quizzes!")
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    if "quiz_notes" not in st.session_state:
        st.session_state.quiz_notes = {
            "answer": None,
            "correct_answer": None,
            "question": None,
            "options": [],
            "submitted": False
        }
    if "quiz_notes_answer" not in st.session_state:
        st.session_state.quiz_notes_answer = None
    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = {
            "answer": None,
            "correct_answer": None,
            "question": None,
            "options": [],
            "submitted": False
        }
    if "file_content" not in st.session_state:
        st.session_state.file_content = ""
    if "progress" not in st.session_state:
        st.session_state.progress = load_progress()
    # Sidebar for navigation
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose a feature", ["Chat", "Study Plan", "File Q&A", "Quiz from Notes", "Quiz from Topic", "Progress Tracker"])
    # Chat Interface
    if app_mode == "Chat":
        st.header("Chat with Study Buddy")
        for message in st.session_state.chat_history.messages:
            with st.chat_message("user" if message.type == "human" else "assistant"):
                st.write(message.content)
        user_input = st.chat_input("Ask a question:")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            try:
                response = answer_chain.invoke(
                    {"question": user_input, "chat_history": format_chat_history(st.session_state.chat_history)},
                    config={"configurable": {"session_id": "default_session"}}
                ).content
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.add_user_message(user_input)
                st.session_state.chat_history.add_ai_message(response)
            except google_exceptions.InvalidArgument as e:
                st.error(f"API key error: {str(e)}. Please verify your Gemini API key.")
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")
    # Study Plan
    elif app_mode == "Study Plan":
        st.header("Create a Study Plan")
        topic = st.text_input("Enter the topic you want to study:")
        time = st.number_input("Enter the time available (in hours):", min_value=0.5, max_value=24.0, step=0.5)
        if st.button("Generate Study Plan"):
            if topic and time:
                try:
                    study_plan = study_plan_chain.invoke(
                        {"topic": topic, "time": str(time), "chat_history": format_chat_history(st.session_state.chat_history)},
                        config={"configurable": {"session_id": "default_session"}}
                    ).content
                    st.write("### Your Study Plan")
                    st.write(study_plan)
                    st.session_state.chat_history.add_user_message(f"Generated study plan for {topic} with {time} hours")
                    st.session_state.chat_history.add_ai_message(study_plan)
                    st.session_state.progress["study_plans"] += 1
                    save_progress()
                except google_exceptions.InvalidArgument as e:
                    st.error(f"API key error: {str(e)}. Please verify your Gemini API key.")
                except Exception as e:
                    st.error(f"Error generating study plan: {str(e)}")
            else:
                st.error("Please provide a topic and time.")
    # File Q&A
    elif app_mode == "File Q&A":
        st.header("Upload Lecture Notes and Ask Questions")
        uploaded_file = st.file_uploader("Upload a PDF, Word, or text file", type=["pdf", "docx", "txt"])
        if uploaded_file:
            st.session_state.file_content = extract_file_text(uploaded_file)
            if "Error" in st.session_state.file_content or "Unsupported" in st.session_state.file_content:
                st.error(st.session_state.file_content)
            else:
                st.success("File uploaded successfully! Ask a question about its content.")
                if len(st.session_state.file_content) > 4000:
                    st.warning("File content exceeds 4000 characters and has been truncated for processing.")
        for message in st.session_state.chat_history.messages:
            if message.type == "human" and "File question" in message.content:
                with st.chat_message("user"):
                    st.write(message.content.replace("File question: ", ""))
                with st.chat_message("assistant"):
                    st.write(st.session_state.chat_history.messages[st.session_state.chat_history.messages.index(message) + 1].content)
        file_question = st.chat_input("Ask a question about the lecture notes:")
        if file_question and st.session_state.file_content:
            if "Error" not in st.session_state.file_content and "Unsupported" not in st.session_state.file_content:
                try:
                    response = file_answer_chain.invoke(
                        {"question": file_question, "file_content": st.session_state.file_content[:4000], "chat_history": format_chat_history(st.session_state.chat_history)},
                        config={"configurable": {"session_id": "default_session"}}
                    ).content
                    with st.chat_message("user"):
                        st.write(file_question)
                    with st.chat_message("assistant"):
                        st.write(response)
                    st.session_state.chat_history.add_user_message(f"File question: {file_question}")
                    st.session_state.chat_history.add_ai_message(response)
                except google_exceptions.InvalidArgument as e:
                    st.error(f"API key error: {str(e)}. Please verify your Gemini API key.")
                except Exception as e:
                    st.error(f"Error answering file question: {str(e)}")
            else:
                st.error("Cannot answer questions due to file extraction error.")
        elif file_question and not st.session_state.file_content:
            st.error("Please upload a file first.")
    # Quiz from Notes
    elif app_mode == "Quiz from Notes":
        st.header("Quiz from Lecture Notes")
        uploaded_file = st.file_uploader("Upload a PDF, Word, or text file for quiz", type=["pdf", "docx", "txt"], key="quiz_file")
        if uploaded_file:
            st.session_state.file_content = extract_file_text(uploaded_file)
            if "Error" in st.session_state.file_content or "Unsupported" in st.session_state.file_content:
                st.error(st.session_state.file_content)
            else:
                st.success("File uploaded successfully! Generate a quiz question.")
                if len(st.session_state.file_content) > 4000:
                    st.warning("File content exceeds 4000 characters and has been truncated for processing.")
        if st.button("Generate Quiz Question", key="generate_quiz_notes"):
            if st.session_state.file_content and "Error" not in st.session_state.file_content and "Unsupported" not in st.session_state.file_content:
                try:
                    quiz = file_quiz_chain.invoke(
                        {"file_content": st.session_state.file_content[:4000], "chat_history": format_chat_history(st.session_state.chat_history)},
                        config={"configurable": {"session_id": "default_session"}}
                    ).content
                    lines = quiz.split("\n")
                    if len(lines) < 6:
                        st.error("Error: Quiz format is invalid. Please try again.")
                    else:
                        question = lines[0].replace("Question: ", "")
                        options = lines[1:5]
                        correct_answer = lines[5].replace("Correct Answer: ", "")
                        st.session_state.quiz_notes["question"] = question
                        st.session_state.quiz_notes["correct_answer"] = correct_answer
                        st.session_state.quiz_notes["options"] = options
                        st.session_state.quiz_notes["answer"] = None
                        st.session_state.quiz_notes["submitted"] = False
                        st.session_state.quiz_notes_answer = None
                except google_exceptions.InvalidArgument as e:
                    st.error(f"API key error: {str(e)}. Please verify your Gemini API key.")
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
            else:
                st.error("Please upload a valid file before generating a quiz.")
        # Display quiz question and answer options
        if st.session_state.quiz_notes["question"]:
            st.write("### Quiz Question")
            st.write(st.session_state.quiz_notes["question"])
            st.session_state.quiz_notes_answer = st.radio(
                "Select an answer:",
                st.session_state.quiz_notes["options"],
                key="quiz_radio_notes",
                index=None
            )
        # Submit answer
        if st.session_state.quiz_notes["question"] and not st.session_state.quiz_notes["submitted"]:
            if st.button("Submit Answer", key="submit_quiz_notes"):
                if st.session_state.quiz_notes_answer:
                    st.session_state.quiz_notes["answer"] = st.session_state.quiz_notes_answer
                    st.session_state.quiz_notes["submitted"] = True
                    is_correct = st.session_state.quiz_notes["answer"] == st.session_state.quiz_notes["correct_answer"]
                    if is_correct:
                        st.success("Correct!")
                        st.session_state.progress["quizzes_correct"] += 1
                    else:
                        st.error(f"Incorrect. The correct answer is: {st.session_state.quiz_notes['correct_answer']}")
                    st.session_state.progress["quizzes_attempted"] += 1
                    st.session_state.progress["quiz_results"].append({
                        "question": st.session_state.quiz_notes["question"],
                        "user_answer": st.session_state.quiz_notes["answer"],
                        "correct_answer": st.session_state.quiz_notes["correct_answer"],
                        "correct": is_correct
                    })
                    st.session_state.chat_history.add_user_message(f"Quiz from notes: {st.session_state.quiz_notes['question']}")
                    st.session_state.chat_history.add_ai_message(f"User answered: {st.session_state.quiz_notes['answer']}, Correct: {st.session_state.quiz_notes['correct_answer']}")
                    save_progress()
                else:
                    st.error("Please select an answer before submitting.")
        # Reset quiz for new question
        if st.session_state.quiz_notes["submitted"]:
            if st.button("Generate New Question", key="new_quiz_notes"):
                st.session_state.quiz_notes = {
                    "answer": None,
                    "correct_answer": None,
                    "question": None,
                    "options": [],
                    "submitted": False
                }
                st.session_state.quiz_notes_answer = None
    # Quiz from Topic
    elif app_mode == "Quiz from Topic":
        st.header("Take a Quiz")
        topic = st.text_input("Enter the topic for the quiz:")
        if st.button("Generate Quiz Question", key="generate_quiz_topic"):
            if topic:
                try:
                    quiz = quiz_chain.invoke(
                        {"topic": topic, "chat_history": format_chat_history(st.session_state.chat_history)},
                        config={"configurable": {"session_id": "default_session"}}
                    ).content
                    lines = quiz.split("\n")
                    if len(lines) < 6:
                        st.error("Error: Quiz format is invalid. Please try again.")
                    else:
                        question = lines[0].replace("Question: ", "")
                        options = lines[1:5]
                        correct_answer = lines[5].replace("Correct Answer: ", "")
                        st.session_state.quiz_topic["question"] = question
                        st.session_state.quiz_topic["correct_answer"] = correct_answer
                        st.session_state.quiz_topic["options"] = options
                        st.session_state.quiz_topic["answer"] = None
                        st.session_state.quiz_topic["submitted"] = False
                        st.write("### Quiz Question")
                        st.write(question)
                        st.session_state.quiz_topic["answer"] = st.radio("Select an answer:", options, key="quiz_radio_topic")
                except google_exceptions.InvalidArgument as e:
                    st.error(f"API key error: {str(e)}. Please verify your Gemini API key.")
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")
            else:
                st.error("Please enter a topic to generate a quiz.")
        if st.session_state.quiz_topic["question"] and not st.session_state.quiz_topic["submitted"]:
            if st.button("Submit Answer", key="submit_quiz_topic"):
                if st.session_state.quiz_topic["answer"]:
                    st.session_state.quiz_topic["submitted"] = True
                    is_correct = st.session_state.quiz_topic["answer"] == st.session_state.quiz_topic["correct_answer"]
                    if is_correct:
                        st.success("Correct!")
                        st.session_state.progress["quizzes_correct"] += 1
                    else:
                        st.error(f"Incorrect. The correct answer is: {st.session_state.quiz_topic['correct_answer']}")
                    st.session_state.progress["quizzes_attempted"] += 1
                    st.session_state.progress["quiz_results"].append({
                        "question": st.session_state.quiz_topic["question"],
                        "user_answer": st.session_state.quiz_topic["answer"],
                        "correct_answer": st.session_state.quiz_topic["correct_answer"],
                        "correct": is_correct
                    })
                    st.session_state.chat_history.add_user_message(f"Quiz on {topic}: {st.session_state.quiz_topic['question']}")
                    st.session_state.chat_history.add_ai_message(f"User answered: {st.session_state.quiz_topic['answer']}, Correct: {st.session_state.quiz_topic['correct_answer']}")
                    save_progress()
                else:
                    st.error("Please select an answer before submitting.")
    # Progress Tracker
    elif app_mode == "Progress Tracker":
        st.header("Your Progress")
        progress = st.session_state.progress
        st.write(f"**Study Plans Generated**: {progress['study_plans']}")
        st.write(f"**Quizzes Attempted**: {progress['quizzes_attempted']}")
        st.write(f"**Quizzes Correct**: {progress['quizzes_correct']}")
        if progress['quizzes_attempted'] > 0:
            accuracy = (progress['quizzes_correct'] / progress['quizzes_attempted']) * 100
            st.write(f"**Quiz Accuracy**: {accuracy:.2f}%")
        else:
            st.write("**Quiz Accuracy**: No quizzes attempted yet.")
        if progress["quiz_results"]:
            st.subheader("Quiz History")
            for i, result in enumerate(progress["quiz_results"], 1):
                st.write(f"**Quiz {i}**:")
                st.write(f"- Question: {result['question']}")
                st.write(f"- Your Answer: {result['user_answer']}")
                st.write(f"- Correct Answer: {result['correct_answer']}")
                st.write(f"- Result: {'Correct' if result['correct'] else 'Incorrect'}")
                st.write("---")
        else:
            st.write("No quiz history available yet.")

if __name__ == "__main__":
    main()