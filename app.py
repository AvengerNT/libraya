import streamlit as st
import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---------------------------------------------
# File paths
# ---------------------------------------------
BOOK_FILE = "book.json"
USER_FILE = "users.json"

# ---------------------------------------------
# Load data
# ---------------------------------------------
def load_books():
    if os.path.exists(BOOK_FILE):
        with open(BOOK_FILE, "r") as f:
            return json.load(f)
    return []

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

books = load_books()
users = load_users()

# ---------------------------------------------
# Recommendation System using TF-IDF
# ---------------------------------------------
def get_recommendations(title, books, top_n=5):
    titles = [book['title'] for book in books]
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(titles)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(range(len(titles)), index=titles)
    idx = indices.get(title, None)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    book_indices = [i[0] for i in sim_scores]
    return [titles[i] for i in book_indices]

# ---------------------------------------------
# Streamlit App
# ---------------------------------------------
st.set_page_config(page_title="Library Management System", layout="wide")

if 'user' not in st.session_state:
    st.session_state.user = None

menu = st.sidebar.selectbox("Menu", ["Login", "Signup", "View Books", "Search & Recommend", "Borrowed Books"])

# ---------------------------------------------
# Login Page
# ---------------------------------------------
if menu == "Login":
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        for user in users:
            if user['username'] == username and user['password'] == password:
                st.session_state.user = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
        else:
            st.error("Invalid credentials.")

# ---------------------------------------------
# Signup Page
# ---------------------------------------------
elif menu == "Signup":
    st.title("📝 Signup")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    
    if st.button("Create Account"):
        if any(u['username'] == username for u in users):
            st.error("Username already exists.")
        else:
            users.append({"username": username, "password": password, "borrowed": []})
            save_users(users)
            st.success("Account created successfully! Please login.")

# ---------------------------------------------
# View Books
# ---------------------------------------------
elif menu == "View Books":
    st.title("📚 All Books")
    df = pd.DataFrame(books)
    st.dataframe(df)

# ---------------------------------------------
# Search and Recommendation
# ---------------------------------------------
elif menu == "Search & Recommend":
    st.title("🔍 Search & Get Recommendations")
    search = st.text_input("Enter Book Title")
    if search:
        found = [b for b in books if search.lower() in b['title'].lower()]
        if found:
            st.success("Book Found")
            for b in found:
                st.write(f"**{b['title']}** by {b['author']} ({b['year']})")
                if st.session_state.user:
                    if st.button(f"Borrow '{b['title']}'", key=b['title']):
                        for user in users:
                            if user['username'] == st.session_state.user:
                                if b['title'] not in user['borrowed']:
                                    user['borrowed'].append(b['title'])
                                    save_users(users)
                                    st.success(f"You borrowed '{b['title']}'")
                                else:
                                    st.info("You already borrowed this book.")

            st.subheader("🤖 Recommended Books")
            recs = get_recommendations(found[0]['title'], books)
            for rec in recs:
                st.write(f"- {rec}")
        else:
            st.warning("No book found with that title.")

# ---------------------------------------------
# Borrowed Books
# ---------------------------------------------
elif menu == "Borrowed Books":
    st.title("📦 Your Borrowed Books")
    if st.session_state.user:
        for user in users:
            if user['username'] == st.session_state.user:
                if user['borrowed']:
                    for book in user['borrowed']:
                        st.write(f"✅ {book}")
                else:
                    st.info("You haven't borrowed any books.")
    else:
        st.warning("Please login to view your borrowed books.")
