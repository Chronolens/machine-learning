import os
import psycopg
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv


class EnvVars:
    def __init__(self):
        self.nats_endpoint = os.getenv("NATS_ENDPOINT")
        self.database_host = os.getenv("DATABASE_HOST")
        self.database_user = os.getenv("DATABASE_USERNAME")
        self.database_password = os.getenv("DATABASE_PASSWORD")
        self.database_port = os.getenv("DATABASE_PORT")
        self.database_name = os.getenv("DATABASE_NAME")


def get_rows_from_db(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id, embedding FROM face_data")
        rows = cursor.fetchall()
        return rows


def cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    embeddings = StandardScaler().fit_transform(embeddings)
    
    cosine_distance = cosine_distances(embeddings)
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    db.fit(cosine_distance)
    
    return db.labels_


def connect_to_database(envs):
    url = f"postgresql://{envs.database_user}:{envs.database_password}@" \
          f"{envs.database_host}:{envs.database_port}/{envs.database_name}"
    try:
        conn = psycopg.connect(url)
        return conn
    except Exception as e:
        raise e


def write_clusters_to_db(conn, ids, labels):
    with conn.cursor() as cursor:
        for id_, label in zip(ids, labels):
            cursor.execute(
                "UPDATE face_data SET cluster_id = %s WHERE id = %s",
                (int(label), int(id_))
            )
        conn.commit() 


def main():
    load_dotenv()
    envs = EnvVars()
    conn = connect_to_database(envs)
    
    rows = get_rows_from_db(conn)
    
    if not rows:
        print("No embeddings found")
        return
    
    ids, embeddings = zip(*rows)
    
    embeddings = np.array([
        np.fromstring(embedding.strip("[]"), dtype=np.float32, sep=",")
        for embedding in embeddings
    ])
    
    labels = cluster_embeddings(embeddings)
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            print(f"Outliers: {np.sum(labels == label)}")
        else:
            print(f"Cluster {label}: {np.sum(labels == label)}")
    
    write_clusters_to_db(conn, ids, labels)
    
    conn.close()


if __name__ == "__main__":
    main()
