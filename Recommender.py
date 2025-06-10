import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

#soup function
def create_soup(x):
    def get_list(data):
        try:
            return [d['name'].replace(" ", "").lower() for d in eval(data)]
        except:
            return []
    genres = get_list(x.get('genres', ''))
    keywords = get_list(x.get('keywords', ''))
    cast = get_list(x.get('cast', ''))[:3]
    director = ''
    try:
        crew_list = eval(x['crew']) if isinstance(x['crew'], str) else []
        director = [member['name'].replace(" ", "").lower() for member in crew_list if member['job'] == 'Director']
    except:
        director = []

    return ' '.join(keywords*2+genres*3+cast*1+director*1)

#cache heavy computation and file loading
@st.cache_data
def load_and_process_data():
    #load CSVs
    movies = pd.read_csv('movies_metadata.csv', low_memory=False)
    credits = pd.read_csv('credits.csv')
    keywords = pd.read_csv('keywords.csv')

    #Convert 'id' to same type
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce')
    
    movies = movies.dropna(subset=['id'])
    credits = credits.dropna(subset=['id'])
    keywords = keywords.dropna(subset=['id'])
    
    movies['id'] = movies['id'].astype(int)
    credits['id'] = credits['id'].astype(int)
    keywords['id'] = keywords['id'].astype(int)
    
    #combine to one source
    movies = movies.merge(credits, on='id')
    movies = movies.merge(keywords, on='id')
    
    #remove unneeded columns
    movies = movies[['title', 'overview', 'genres', 'cast', 'crew', 'keywords']]
    movies = movies.reset_index(drop=True)
    
    #apply the soup
    movies['soup'] = movies.apply(create_soup, axis=1)

    #TF-IDF and cosine similarity
    vectorizer = CountVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(movies['soup'])
    
    #Cosine similarity
    cosine_sim = cosine_similarity(matrix, matrix)
    
    #index of titles
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    return movies, cosine_sim, indices 

#recommendation Function
def recommend(title, cosine_sim, indices, movies, n=10):
    try:
        idx = indices[title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
    except KeyError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = [
        score for score in sim_scores
        if score[0] != idx and float(score[1]) > 0
    ]
    sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)[:n]
    movie_indices = [i for i, _ in sim_scores if i < len(movies)]

    return movies.loc[movie_indices, 'title'].tolist()
#-------------------------- Streamlit App ----------------------------#

st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Movie Recommender App")

movies, cosine_sim, indices = load_and_process_data()

if indices.empty:
    st.error("No movies loaded. Please check your CSV files.")
    st.stop()


#Movie selection UI
movie_list = sorted([title for title in indices.index if isinstance(title, str)])

selected_movie = st.selectbox("Pick a movie you like:", movie_list)
num_recs = st.slider("How many recommendations?", min_value=5, max_value = 100, value=20)

if 'recs' not in st.session_state:
    st.session_state.recs = []
if 'prev_movie' not in st.session_state:
    st.session_state.prev_movie = None
if 'prev_num_recs' not in st.session_state:
    st.session_state.prev_num_recs = None

if selected_movie != st.session_state.prev_movie or num_recs != st.session_state.prev_num_recs:
    st.session_state.recs = []
    st.session_state.prev_movie = selected_movie
    st.session_state.prev_num_recs = num_recs

if st.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        st.session_state.recs = recommend(selected_movie, cosine_sim, indices, movies, n=num_recs)

if st.session_state.recs:
    st.subheader(f"Top {num_recs} movies similar to *{selected_movie}*:")
    for i, rec in enumerate(st.session_state.recs, 1):
        st.write(f"{i}. {rec}")

st.write("Soup for selected movie:")
st.write(movies.loc[indices[selected_movie], 'soup'])
