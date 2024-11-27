import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tabulate import tabulate

class DataEntry:
    title: str
    artists: str
    album: str
    track_id: str
    
    def __init__(self, title, artists, album, track_id):
        self.title = title
        self.artists = artists
        self.album = album
        self.track_id = track_id

if 'data_load_done' not in st.session_state:    
    st.session_state.data_load_done = False
    
if 'data' not in st.session_state:    
    st.session_state.data = pd.read_csv("./data/dataset.csv", usecols=['energy', 'valence', 'tempo', 'danceability', 'speechiness', 'track_id', 'track_name', 'artists', 'album_name'])
    # Cutting the Genre and removing duplicates
    st.session_state.data = st.session_state.data.drop_duplicates()    
    st.session_state.data = st.session_state.data.sample(frac=1)   
    st.session_state.data_classes = [DataEntry(a.track_name,a.artists,a.album_name,a.track_id) for a in st.session_state.data.itertuples()]
    st.session_state.data_load_done = True
 
def run_k_nearest_algorithm(form, input_song):
    # max songs 89741
    song_count = 89741
    # Generate Subset with song_count and needed columns
    subset_with_id_and_name = st.session_state.data
    subset = subset_with_id_and_name[['energy','valence','tempo','danceability','speechiness']]
    
    # extract input_song from data set
    #input_song_id = st.session_state.data.loc[st.session_state.data["track_id"] == input_song.track_id]
    input_song_index = subset_with_id_and_name.loc[st.session_state.data["track_id"] == input_song.track_id].index
    # start_value = input_song_id[['energy', 'valence', 'tempo', 'danceability', 'speechiness']]
    
    # Normalize Data
    norm_subset = subset
    norm_subset = (norm_subset - norm_subset.min()) / (norm_subset.max() - norm_subset.min())
    X_train_norm = norm_subset
    
    
    X_train_norm.insert(1, 'org_index', subset_with_id_and_name.index) 
    start_value = X_train_norm.loc[input_song_index]
    
    # Run Nearest Neighbour
    samples = X_train_norm

    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(samples)
    neighbour_indizes = neigh.kneighbors(start_value.to_numpy(), 10, return_distance=False)
    
    # Extract Songs 
    print_table_kNearest = []
    value_list_kNearest = []
    org_indices_kNearest = []

    # Get original indices
    for index in neighbour_indizes:
        org_indices_kNearest.append(subset_with_id_and_name.iloc[index].index)

    org_indices_kNearest = org_indices_kNearest[0]

    for index in org_indices_kNearest:
        print_table_kNearest.append(DataEntry(subset_with_id_and_name.loc[index].track_name, subset_with_id_and_name.loc[index].artists, subset_with_id_and_name.loc[index].album_name, subset_with_id_and_name.loc[index].track_id))
        value_list_kNearest.append([X_train_norm.loc[index].energy, X_train_norm.loc[index].valence, X_train_norm.loc[index].tempo, X_train_norm.loc[index].danceability, X_train_norm.loc[index].speechiness])
    
    #df = pd.DataFrame(print_table_kNearest, columns=["Artists"])
    # print outcome
    #st.list(df)
    for entry in print_table_kNearest:
        st.markdown(str(entry.artists) + " | " + str(entry.title) + ", Album: " + str(entry.album))

    #form.write("\n\nKNearest Neighbours")
    #form.write(tabulate(print_table_kNearest, headers=['Artists','Trackname']))  
    

def generate_playlist(input_song, form):
    if(input_song != None):
        run_k_nearest_algorithm(form, input_song)


# Initializing Streamlit App
st.title("Insane Song Recommender")

with st.form("song_input"):
    #song_input = st.multiselect(options=st.session_state.data["track_name"], label="Input Song pls", max_selections=1)
    song_input = st.selectbox(options=st.session_state.data_classes, label="Input Song pls",placeholder="Search for a Song", index=None, key="song_input_widget", format_func=lambda entry: str(entry.artists) + " | " + str(entry.title) + " | " + str(entry.album))
    st.form_submit_button("Generate Playlist", on_click=generate_playlist(song_input, st))
    #st.form_submit_button("Generate Playlist", on_click=st.write(song_input.track_id))
    
    
   
