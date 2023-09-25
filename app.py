import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

# from functions import *


st.set_page_config(layout="centered")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📚Book Recommendation Web App")
st.markdown("#")
st.markdown("#")


st.sidebar.markdown(f" ## :gear: Recommendation Settings")
st.sidebar.markdown("---")
no_of_rec = int(st.sidebar.slider("Select Number of Book Recommendations", 1, 50, 10))
n_cols = st.sidebar.number_input("Select Number of columns", 5)
n_cols = int(n_cols)


@st.cache_resource
def load_data():
    df = pd.read_csv("filtered_df.csv")

    book_titles = pickle.load(open("unique_book_titles.pkl", "rb"))
    user_ids = pickle.load(open("unique_user_ids.pkl", "rb"))

    decoded_titles = [title.decode("utf-8") for title in book_titles]
    decoded_user_ids = [user.decode("utf-8") for user in user_ids]

    # Load model
    loaded_model = tf.saved_model.load("export")

    return decoded_titles, decoded_user_ids, loaded_model, df


unique_book_titles, unique_user_ids, rec_model, df = load_data()


def recommend_books(user_id, top_k):
    recommendations = []
    ratings = {}

    for book_title in unique_book_titles[:top_k]:
        ratings[book_title] = rec_model(
            {"user_id": np.array([user_id]), "book_title": np.array([book_title])}
        )

    for title, score in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        top_books = {}
        top_books["title"] = title
        top_books["score"] = f"{score[0][0]: .2f}"
        recommendations.append(top_books)

    return recommendations


def image_cover(df, book_name):
    link = df[df["book_title"] == book_name]["img_l"].values

    if len(link) > 1:
        return link[1]
    else:
        return link[0]


def get_user(df, id):
    # books = ""
    user_data = df[df["user_id"] == id][:5]
    books = user_data["book_title"].values
    rating = user_data["rating"].values
    authors = user_data["book_author"].values

    return books, rating, authors


user_id = st.selectbox("Select a user", unique_user_ids)
rec_btn = st.button("Recommend Books")
st.markdown("#")
st.markdown("#")


plc_holder = st.container()


if rec_btn:
    with plc_holder:
        st.markdown(f"#### These are some of the books user {user_id} has read")
        st.markdown("---")
        books, ratings, authors = get_user(df, int(user_id))

        n_rows = int(1 + 3 // 3)
        rows = [st.columns(n_cols) for _ in range(3)]
        cols = [column for row in rows for column in row]

        for col, title, rating, author in zip(cols, books, ratings, authors):
            col.write(f" :blue[Title]: {title[:15]}...")
            col.write(f" :blue[Rating]: {rating}")
            col.write(f" :blue[Author]: {author}")
            col.image(image_cover(df, title))
    st.markdown("---")

    # RECOMMENDATION SIDE
    st.subheader(f"Top {no_of_rec} Ranked Book Recommendations for user {user_id}")
    st.markdown("---")

    top_rec = recommend_books(user_id, no_of_rec)

    covers = []
    titles = []
    scores = []

    for rec in top_rec:
        covers.append(image_cover(df, rec["title"]))
        titles.append(rec["title"])
        scores.append(rec["score"])

    n_rows = int(1 + no_of_rec // n_cols)
    rows = [st.columns(n_cols) for _ in range(n_cols)]
    cols = [column for row in rows for column in row]

    for col, poster, title, score in zip(cols, covers, titles, scores):
        col.markdown(f"###### :blue[Title]: {title[:15]}...")
        col.write(f" :blue[Rank]: {score}")

        col.image(poster)
