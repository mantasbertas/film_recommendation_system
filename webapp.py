import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import recommendation_system
from bs4 import BeautifulSoup
import random
import time

API_KEY = 'cc7244b918389e2fc71a91a0792ef848'
st.set_page_config(page_title='Movie Recommendation System', layout='wide', initial_sidebar_state='auto')

model_path = 'models/properly_scaled_model.h5'
scaler_path = 'models/scaler.pkl'
movie_database_path = 'data/film_data/prepared_film_data.csv'
user_watchlists_path = 'data/user_ratings_data/user_watchlists.csv'

def scrape_movie(imdb_id):
    st.text(f"Scraping data for movie {imdb_id}...")
    response = requests.get(f'https://api.themoviedb.org/3/find/{imdb_id}?api_key={API_KEY}&external_source=imdb_id')
    data = response.json()

    movie_results = data.get('movie_results', [])
    if movie_results:
        movie = movie_results[0]
        title = movie['title']
        poster_path = movie['poster_path']
        if poster_path:
            poster_link = f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            poster_link = None
        imdb_link = f"https://www.imdb.com/title/{imdb_id}"
        return title, poster_link, imdb_link
    return None, None, None


def fetch(url, retries=5, delay=5, backoff_factor=2):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 403:
                print("Private ratings encountered. Skipping user.")
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if response.status_code != 403:
                sleep_time = delay * (backoff_factor ** attempt) + random.uniform(0, 0.1)
                print(f"Request failed with {e}. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
    print("Failed to fetch the URL after multiple retries. Skipping user.")
    return None


def scrape_user_ratings(user_url):
    user_id = user_url.split("/")[4]
    ratings = []
    user_url += 'ratings'

    while user_url:
        response = fetch(user_url)

        if response is None:
            break

        soup = BeautifulSoup(response.content, "html.parser")

        rating_elements = soup.find_all("div", class_="lister-item")

        if not rating_elements:
            break

        for element in rating_elements:
            try:
                imdb_id = element.find("a", href=lambda href: href and "/title/" in href)["href"].split("/")[2]
                rating_string = element.find_all("span", class_="ipl-rating-star__rating")[1].text.strip()
                rating = int(rating_string)
                ratings.append((user_id, imdb_id, rating))
            except Exception as e:
                print(f"Error occurred while parsing a rating for user {user_id}: {e}. Skipping this rating.")

        next_button = soup.find("a", class_="flat-button lister-page-next next-page")

        if next_button:
            user_url = "https://www.imdb.com" + next_button["href"]
        else:
            user_url = None

    return ratings


def main():
    st.title('🎥 Movie Recommendation System 🎥')
    st.markdown("#### Please paste your IMDb user URL or upload your movie ratings list")

    user_url = st.text_input("Paste your IMDb user URL here")
    uploaded_file = st.file_uploader("Or choose a file")

    watchlist = None

    if user_url:
        st.text("Scraping your ratings from IMDb...")
        ratings = scrape_user_ratings(user_url)


        watchlist = pd.DataFrame(ratings, columns=['user_id', 'imdb_id', 'rating'])

        movie_database = pd.read_csv(movie_database_path)

        watchlist = watchlist[watchlist['imdb_id'].isin(movie_database['tconst'])]
    elif uploaded_file is not None:
        watchlist = pd.read_csv(uploaded_file)

    if watchlist is not None:
        st.text("Preparing your recommendations... this might take a few minutes")
        recommendations = recommendation_system.movie_recommendation_pipeline(watchlist, user_watchlists_path, movie_database_path, scaler_path, model_path)
        movie_data = []

        for _, row in recommendations.iterrows():
            imdb_id = row['imdb_id']
            predicted_rating = row['predicted_rating']
            title, poster_link, imdb_link = scrape_movie(imdb_id)
            if title and poster_link and imdb_link:
                movie_data.append((title, poster_link, imdb_link, predicted_rating))

        movie_data = sorted(movie_data, key=lambda x: x[3], reverse=True)[:10]
        st.success("Here are your movie recommendations!")

        cols = st.columns(5)
        for i in range(2):
            for j in range(5):
                index = i*5 + j
                if index < len(movie_data):
                    title, poster_link, imdb_link, predicted_rating = movie_data[index]
                    if poster_link:
                        cols[j].image(Image.open(BytesIO(requests.get(poster_link).content)), width=200, caption=f"[{title}]({imdb_link})\nPredicted Rating: {predicted_rating}")

if __name__ == "__main__":
    main()
