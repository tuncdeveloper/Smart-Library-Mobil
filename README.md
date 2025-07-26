# 📚 Smart Library Mobile

Smart Library Mobile is a modern mobile library application designed to enhance the book reading experience. 
Users can log in, browse books, add them to their favorites, update personal information, and discover new books through an AI-powered recommendation system.

---

## 🚀 Features

- 🔐 **Secure authentication using JWT**
- 📚 **Browse books**, filter by category
- ❤️ **Add books to favorites**
- 🤖 **AI-powered book recommendation system** (based on user's favorites)
- 🧠 **Content similarity analysis with NLP models** (embedding or cosine similarity)
- 🛠️ **User profile update functionality**
- 📱 Built with modern React Native UI
- 🌐 Robust backend integration using Spring Boot RESTful API

---

## 🧠 AI-Powered Book Recommendation System

The application analyzes users' favorite books to provide personalized recommendations based on content similarity.  
The model is trained on book titles, descriptions, and categories. Suggestions are served via a Python-based recommendation service.

---

## 🔄 Backend - Model Service Integration

The Spring Boot backend communicates with the recommendation system using `WebClient`. This reactive and non-blocking architecture ensures efficient and scalable service communication.



NOT : Some of the libraries used during the training of the AI model are 
located in the .venv folder. However this folder 
has not been included in the Git repository due to its large size