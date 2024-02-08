import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) > 1:
    DATADIR = sys.argv[1]
else:
    # Default DATADIR directory if not provided via command-line arguments
    DATADIR = "."

raw_temp_dir = os.path.join(DATADIR, "dataTemp/raw_temp")
raw_dir = os.path.join(DATADIR, "data/raw")
curated_dir = os.path.join(DATADIR, "data/curated")
training_dir = os.path.join(DATADIR, "data/training")
log_dir = os.path.join(DATADIR, "Log")
statistics_dir = os.path.join(DATADIR, "data/statistics")


def read_data(dir, selection):
    try :
        df = pd.read_csv(os.path.join(dir, selection), sep =";", encoding='Windows-1252')
        if len(df.columns) == 1:
            df = pd.read_csv(os.path.join(dir, selection),sep =",", encoding='Windows-1252')
    except Exception:
        df = pd.read_csv(os.path.join(dir, selection), sep =";", encoding='utf-8')
        if len(df.columns) == 1:
            df = pd.read_csv(os.path.join(dir, selection),sep =",", encoding='utf-8')

    return df

# Page d'upload de fichier CSV
def upload_csv_page():
    st.title("Uploader un fichier CSV dans raw_temp")
    uploaded_file = st.file_uploader("Uploader un fichier CSV", type=['csv'])
    if uploaded_file:
        with open(os.path.join(raw_temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success("Fichier CSV uploadé avec succès dans raw_temp")

def display_png(image_path, key=None):
    if os.path.exists(image_path):
        st.image(image_path, caption="Image PNG", use_column_width=True, key=key)
    else:
        st.error("L'image spécifiée n'existe pas.")

# Page pour afficher les CSV disponibles
# Page pour afficher les CSV disponibles
def display_csv_page():
    st.title("choose pipeline version")
    st.subheader("select a version")
    files = sorted([f[-14:-4] for f in os.listdir(raw_dir) if f.lower().endswith('.csv')])
    selected_pipeline_version = st.selectbox("select a pipeline versioned with timestamp", files)

    
    st.subheader("raw")
    if selected_pipeline_version:
        raw_csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(f'{selected_pipeline_version}.csv')][0]
        df_raw = read_data(raw_dir, raw_csv_files)
        st.write(df_raw)

        st.subheader("curated")
        curated_csv_files = [f for f in os.listdir(curated_dir) if f.lower().endswith(f'{selected_pipeline_version}.csv')][0]
        df_curated = read_data(curated_dir, curated_csv_files)
        st.write(df_curated)
    
        st.subheader("training")
        training_csv_files = [f for f in os.listdir(training_dir) if f.lower().endswith(f'{selected_pipeline_version}.csv')][0]
        df_training = read_data(training_dir, training_csv_files)
        st.write(df_training)

        st.subheader("statistics")
        stats_csv_files = [f for f in os.listdir(statistics_dir) if f.lower().endswith(".csv")][0]
        df_stats = read_data(statistics_dir, stats_csv_files)
        df_confusion_matrix = df_stats[df_stats.model_timestamp.str.contains(selected_pipeline_version)]
        st.dataframe(df_confusion_matrix)
        #print(type(df_confusion_matrix.cm.values[0]))
        cm_values = [int(x) for x in df_confusion_matrix.cm.values[0].strip('[]').replace('[', '').replace(']', '').split()]

       
        # Reshape the list of values into a 2D matrix
        try :
            confusion_matrix = np.array(cm_values).reshape((2, 2))  
                
            print(confusion_matrix)  # Remove the outer brackets

            # Reshape the list of values into a 2D matrix
        
            display_confusion_matrix(confusion_matrix)
        except Exception:
            pass

        

    else:
        st.write("Aucun fichier CSV avec un timestamp trouvé dans raw.")


# Page pour afficher les logs
def display_logs_page():
    st.title("Afficher les log")
    log_file_path = os.path.join(log_dir, "pipeline_log.txt")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            logs = log_file.read()
            st.text_area("Logs du fichier pipeline_log.txt", value=logs, height=500)
    else:
        st.error("Le fichier de logs n'existe pas")

def display_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    img = ax.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")

    st.pyplot(fig)


# Navigation entre les pages
pages = {
    "Uploader un fichier CSV": upload_csv_page,
    "choose pipeline version": display_csv_page,
    "Afficher les logs": display_logs_page,
    #"Afficher les statistiques du modèle": display_statistics_page
}

selected_page = st.sidebar.selectbox("Sélectionner une page", list(pages.keys()))
pages[selected_page]()
