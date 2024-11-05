import os
import psycopg
import numpy as np
import logging
import time 

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cluster_service")


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
        logger.info("Fetching rows from the database.")
        cursor.execute("SELECT id, embedding FROM face_data")
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} rows.")
        return rows


def cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    logger.info("Standardizing embeddings and calculating cosine distances.")
    embeddings = StandardScaler().fit_transform(embeddings)
    cosine_distance = cosine_distances(embeddings)

    logger.info("Running DBSCAN clustering.")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    db.fit(cosine_distance)

    logger.info(f"Clustering completed. Number of clusters found: {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}.")
    return db.labels_


def connect_to_database(envs):
    url = f"postgresql://{envs.database_user}:{envs.database_password}@" \
          f"{envs.database_host}:{envs.database_port}/{envs.database_name}"
    try:
        logger.info("Attempting to connect to the database.")
        conn = psycopg.connect(url)
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        logger.error("Failed to connect to the database.", exc_info=True)
        raise e


def write_clusters_to_db(conn, ids, labels):
    with conn.cursor() as cursor:
        logger.info("Writing clusters to the database.")
        for id_, label in zip(ids, labels):
            cursor.execute(
                "UPDATE face_data SET cluster_id = %s WHERE id = %s",
                (int(label), int(id_))
            )
        conn.commit()
        logger.info("Clusters successfully written to the database.")



def main():

    load_dotenv()
    logger.info("Loaded environment variables.")
    
    envs = EnvVars()
    
    while True:
        conn = connect_to_database(envs)

        rows = get_rows_from_db(conn)

        if not rows:
            logger.info("No embeddings found in the database.")
            conn.close()
            logger.info("Database connection closed.")
            time.sleep(3600)
            continue

        ids, embeddings = zip(*rows)

        embeddings = np.array([
            np.fromstring(embedding.strip("[]"), dtype=np.float32, sep=",")
            for embedding in embeddings
        ])

        labels = cluster_embeddings(embeddings)

        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label == -1:
                logger.info(f"Outliers: {np.sum(labels == label)}")
            else:
                logger.info(f"Person {label}: Identified photos - {np.sum(labels == label)}")

        write_clusters_to_db(conn, ids, labels)

        conn.close()
        logger.info("Database connection closed.")

        logger.info("60 Minutes until next clustering cycle.")
        time.sleep(3600)

if __name__ == "__main__":
    main()
