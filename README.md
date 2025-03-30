# 🤖 Sales Chatbot

## 📌 Overview
This project is an AI-powered chatbot that allows users to query sales data using natural language. It utilizes **FastAPI**, **LangChain**, and **Google Gemini AI** to process queries and return insights from a sales dataset.

## 🚀 Features
- 🔍 Query sales data using natural language.
- 🤖 AI-powered Pandas query generation.
- ⚡ FastAPI backend for API handling.
- 🧠 Uses LangChain for LLM integration.
- 💻 Interactive Streamlit frontend for user-friendly querying.

## 🛠 Technologies Used
- 🏗 **FastAPI** - Handles API requests.
- 🔗 **LangChain** - Integrates with LLMs for query generation.
- 🌍 **Google Gemini AI** - Processes natural language queries.
- 📊 **Pandas** - Handles data processing and analysis.
- 🖥 **Streamlit** - Provides an interactive frontend interface.

## ⚙️ Installation & Setup
1. 📥 Clone the repository:
   ```sh
   git clone https://github.com/Sanchitkanwar31/Chatbot_Sales_dataset.git
   cd Chatbot_Sales_dataset
   ```
2. 🔧 Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. ▶️ Run the FastAPI server:
   ```sh
   uvicorn backend.main:app --reload
   ```
4. 🌐 Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

## 🖥 Running the Frontend
1. Navigate to the frontend directory:
   ```sh
   cd frontend
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
3. Open your browser and visit:
   ```
   http://localhost:8501
   ```

## 📝 Usage Example
- **User Input:** "Show total sales for last month."
- **🤖 AI Generates:** A Pandas query to fetch the required data.
- **📊 Response:** Displays the total sales for the specified period.

## Summary of Working
- ![image](https://github.com/user-attachments/assets/5e9041bf-5b14-4c57-a5ff-12914c9f1b8c)


## 🔮 Future Enhancements
- 🎯 Improve AI accuracy with fine-tuned prompts.
- 🔒 Add authentication for secure access.
- 🔍 Integrate FAISS for advanced semantic search.

## 📜 License
This project is open-source and available under the MIT License.

