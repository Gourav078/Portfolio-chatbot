import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_KEY'] = os.getenv("HF_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Load environment variables (if needed for API keys)

# Set up the title and layout
st.set_page_config(page_title="Gourav Mitra Portfolio", page_icon=":guardsman:", layout="wide")

# Title Section
st.markdown(
    """
    <div style="text-align: center; font-size: 40px; font-weight: bold;">Gourav Mitra</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; font-size: 20px;">Ask me anything about my work, projects, or experience.</div>
    """,
    unsafe_allow_html=True
)

# Left sidebar with skills and achievements
with st.sidebar:
    st.header("Skills")

    st.subheader("Languages")
    st.write("Python, SQL (Structured Query Language) Javascript, TypeScript, Dart, Java ")

    st.subheader("Data Science and Machine Learning")
    st.write(
        """
        Data science pipeline (cleansing, wrangling, visualization, modeling, interpretation), Statistics, 
        Machine Learning (CNN, RNN) using Keras, TensorFlow, and NLP techniques with PyTorch.
        """
    )
    st.subheader("Computer Vision")
    st.write("Object Detection and Tracking, Image Processing")

    st.header("Gen AI")
    st.write("Document Chains, History-Aware Retrieval, Prompt Engineering")

    st.subheader("Web Development Skills")
    st.write(
        "React.js, Node.js, Express.js, MongoDB, PostgreSQL, RESTful APIs, TypeScript, JavaScript, Tailwind CSS, Next.js")

    st.subheader("Flutter Development Skills")
    st.write("Flutter, Dart, Mobile App Development, UI/UX Design")

    st.subheader("Backend Development Skills")
    st.write(
        "Node.js, Express.js, MongoDB, PostgreSQL, RESTful APIs, ")

    st.subheader("Soft Skills")
    st.write("Project Management, Teamwork, Algorithms")

# Streamlit setup for the user interface


loader = TextLoader("About Gourav.txt")  # Default text loader
text_documents = loader.load()

# Split the documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(text_documents)

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant specialized in answering questions about Gourav Mitra, his projects, skills and Experience. "
    "When answering, provide detailed and structured answers. If the user asks for Pratham's projects,skills and Experience "
    "list them with names, descriptions, and any related achievements. "
    "If the question is about a specific project, give an elaborative response with relevant details. "
    "If any technical terms are mentioned, explain them in simple terms."
    "\n\n{context}"
)

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history,"
    "and give a elaborative answer according to question "
    "asked and if there are some technical terms explain them"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),

        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever using Langchain
history_aware_retriver = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the question-answering chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriver, question_answer_chain)

# Streamlit input form for user question
user_input = st.text_input("Ask a question:")

# When the user submits a question
if user_input:
    response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})

    # Store the question and answer in chat history
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"])
        ]
    )

    # Display the answer in Streamlit
    st.write("Answer:", response["answer"])

# Optionally, you can also provide a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history.clear()
    st.write("Chat history cleared.")


def download_chat_history():
    chat_text = ""
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            chat_text += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            chat_text += f"AI: {message.content}\n"
    return chat_text


# Add a download button for the chat history
st.download_button(
    label="Download Chat History",
    data=download_chat_history(),
    file_name="chat_history.txt",
    mime="text/plain"
)


# Add a heading for  Projects
st.markdown(
    """
    <div style="text-align: center; font-size: 30px; font-weight: bold;">Projects</div>
    """,
    unsafe_allow_html=True
)

# List of Other Projects with concise descriptions
st.markdown(
    """
        **1. Image Caption Generator using CNN, RNN, and VGG16 – Python, TensorFlow, Keras**:  
    Developed an image caption generator model by combining Convolutional Neural Networks (CNN) with Recurrent Neural Networks (RNN), utilizing VGG16 for feature extraction.  
    Used VGG16 pre-trained on ImageNet to extract high-level features from images, which were then passed through an RNN (LSTM) to generate descriptive captions.  
    Achieved high accuracy in generating meaningful and contextually relevant captions, improving the model's understanding of visual data.  
    Leveraged deep learning techniques like transfer learning (VGG16) and sequence generation (RNN) for the image captioning task.

    **2. AI Chatbot with GROQ, LangChain, and Hugging Face – Python, LangChain, Hugging Face, GROQ**:  
    Developed an advanced AI chatbot leveraging GROQ, LangChain, and Hugging Face to provide natural and conversational interactions.  
    Integrated LangChain to manage conversational workflows, handle user queries, and process responses in an organized and modular way.  
    Utilized Hugging Face's Transformer models for NLP tasks like question answering, text generation, and sentiment analysis to improve response accuracy and context understanding.  
    Employed GROQ to query data in real-time, allowing the chatbot to retrieve dynamic information from structured data sources efficiently.  
    Focused on building a robust, scalable system that can be deployed in both small-scale applications and enterprise-level services, ensuring high responsiveness and real-time interaction.  
    Enhanced the chatbot with multi-turn conversation handling to maintain context across different user inputs and make the chatbot more interactive and intelligent.

    **3. Point Cloud Classification – Python, TensorFlow, PyTorch, 3D Data Processing**:  
    Applied deep learning techniques to classify 3D point cloud data into distinct categories, such as buildings, vehicles, and trees.  
    Utilized neural networks such as PointNet and PointNet++ to process raw 3D point cloud data, achieving high classification accuracy.  
    Enhanced expertise in processing 3D spatial data, leveraging advanced techniques in point cloud segmentation and feature extraction.  
    This project enabled a deeper understanding of complex environments, making it applicable to areas such as autonomous driving, urban planning, and environmental monitoring.

    **4. 3D Point Cloud and Mesh Generation Using a Single Image – Python, OpenCV, Hugging Face, Deep Learning, 3D Reconstruction**:  
    Developed a system to reconstruct 3D point clouds and meshes from a single 2D image, leveraging photogrammetry principles combined with pre-trained 3D models from Hugging Face for realistic 3D representations.  
    Utilized a pre-trained 3D model from Hugging Face, such as MiDaS or DETR3D, to estimate depth maps and 3D structure from 2D images, significantly improving reconstruction quality and reducing training time.  
    Enhanced 3D modeling and mesh generation by integrating depth estimation techniques and multi-view stereo (MVS) algorithms.  
    Enabled applications in 3D modeling, augmented reality (AR), and virtual environments by offering a streamlined pipeline for converting 2D images into rich, detailed 3D content.  
    This approach not only increased the accuracy of the 3D reconstructions but also made it easier to generate complex 3D models from just a single photograph or image, leveraging cutting-edge pre-trained models for fast and efficient processing.

    **5. Machine Learning Real Estate Price Prediction – Python, Pandas, Scikit-learn, Matplotlib**:  
    Developed a Machine Learning model to predict real estate prices using historical data of properties such as location, size, amenities, and market trends.  
    Leveraged Pandas for data preprocessing and feature engineering, cleaning and transforming raw data into a format suitable for machine learning models.  
    Used Scikit-learn to train multiple regression models, including Linear Regression, Decision Trees, and Random Forests, for price prediction.  
    Evaluated model performance using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score to ensure the accuracy of predictions.  
    Visualized key data insights and model predictions using Matplotlib and Seaborn, helping to identify trends, correlations, and outliers in the dataset.  
    Implemented feature selection techniques to reduce overfitting and improve the generalizability of the model.  
    The model provided valuable insights for real estate investors, helping them make informed decisions by predicting property prices based on historical data and market conditions.

    **6. Ecommerce Website – HTML, CSS, JavaScript**:  
    Designed and developed a fully functional e-commerce website focused on enhancing user experience.  
    Achieved a 25% reduction in page load times, resulting in a 15% decrease in bounce rates and a 20% increase in conversion rates.  
    Implemented pricing strategies that led to a 10% increase in average order value compared to competitors.

    **7. Admin Dashboard – React JS, Tailwind CSS, Node JS, Express JS, Mongo DB**:  
    Developed a user-friendly admin dashboard to streamline management tasks and data visualization.  
    Increased admin dashboard usage by 40%, with an average time spent of 30 minutes per session.  
    Improved user interaction with charts by 25% and increased calendar events added/modified by 15%.

    **8. Python Turtle Graphics – Python**:  
    Developed interactive graphical drawings using Python's Turtle module to create various shapes and designs.  
    Utilized Turtle’s drawing capabilities to create dynamic shapes, patterns, and designs with simple commands.  
    Demonstrated proficiency in understanding Python logic and graphical design by creating both static and animated visuals.  
    Enhanced problem-solving skills by applying loops and functions to produce intricate patterns and drawings.

    **9. Employee Registration Form with Tomcat, MySQL, JDBC, Hibernate, Spring, Java7, and Eclipse – Java, MySQL, Hibernate, Spring, JDBC, Tomcat, Eclipse**:  
    Developed an Employee Registration System that allows users to register employees into a database using a web-based interface.  
    - **Backend Technologies**:  
      Implemented JDBC to handle database connections and interactions, allowing CRUD (Create, Read, Update, Delete) operations on the employee data stored in MySQL.  
      Used Hibernate for ORM (Object-Relational Mapping), simplifying the interaction with the MySQL database and ensuring efficient database operations.  
      Integrated Spring Framework to handle dependency injection, provide transaction management, and manage the business logic layer.  
    - **Frontend Technologies**:  
      Designed a user-friendly registration form in the web interface for employees to input their data, such as name, department, and job title.  
    - **Deployment**:  
      Deployed the application on Tomcat, which served as the web server for handling HTTP requests and rendering the registration form.  
    - **Development Environment**:  
      Developed and tested the application using Eclipse IDE and Java 7, utilizing its debugging and integration tools for a smooth development experience.  
    - **Key Features**:  
      Seamless integration between the frontend (HTML forms) and backend (Spring, Hibernate, and MySQL) through MVC architecture.  
      Ensured the system supports scalability and maintains performance through optimized SQL queries and Hibernate configuration.  
      Implemented form validation and error handling to ensure a robust and user-friendly experience.
    """,
unsafe_allow_html=True
)

