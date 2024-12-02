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


def get_rows_from_db(conn):
    with conn.cursor() as cursor:
        logger.info("Fetching rows from the database.")
        cursor.execute("""
            SELECT media_face.id, media_face.embedding, media_face.cluster_id, media.user_id
            FROM media_face
            JOIN media ON media_face.media_id = media.id
        """)
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} rows from the database.")
        return rows


def cluster_embeddings(embeddings, eps=0.5, min_samples=3):
    logger.info("Standardizing embeddings and calculating cosine distances.")
    start_time = time.time()
    embeddings = StandardScaler().fit_transform(embeddings)
    cosine_distance = cosine_distances(embeddings)

    logger.info("Running DBSCAN clustering.")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    db.fit(cosine_distance)
    end_time = time.time()

    logger.info(f"Clustering completed in {end_time - start_time:.2f} seconds. "
                f"Number of clusters found: {len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)}.")
    return db.labels_


def assign_consistent_cluster_ids(existing_data, new_labels, ids, conn, user_id):

    logger.info(f"Assigning consistent cluster IDs for user ID {user_id}.")

    existing_id_to_cluster = {id_: cluster_id for id_, _, cluster_id, _ in existing_data}
    new_cluster_assignments = {}

    for new_cluster in set(new_labels):
        if new_cluster == -1:
            logger.debug("Skipping outlier cluster.")
            continue

        cluster_member_ids = [ids[i] for i, label in enumerate(new_labels) if label == new_cluster]
        matched_cluster_ids = {existing_id_to_cluster[id_] for id_ in cluster_member_ids if existing_id_to_cluster[id_] is not None}

        if matched_cluster_ids:
            assigned_cluster_id = matched_cluster_ids.pop()
            logger.debug(f"Using existing cluster ID {assigned_cluster_id} for this new cluster.")
        else:
            with conn.cursor() as cursor:
                cursor.execute("INSERT INTO cluster (user_id) VALUES (%s) RETURNING id", (user_id,))
                assigned_cluster_id = cursor.fetchone()[0]
                conn.commit()
            logger.debug(f"Assigned new cluster ID {assigned_cluster_id} for user ID {user_id}.")

        for member_id in cluster_member_ids:
            new_cluster_assignments[member_id] = assigned_cluster_id

    consistent_labels = [new_cluster_assignments.get(id_, existing_id_to_cluster.get(id_, -1)) for id_ in ids]
    consistent_labels = [label if label is not None else -1 for label in consistent_labels]
    
    logger.info(f"Consistent cluster IDs assigned for user ID {user_id}.")
    return consistent_labels


def write_clusters_to_db(conn, ids, labels):
    
    with conn.cursor() as cursor:
        logger.info("Writing cluster IDs to the database.")
        for id_, label in zip(ids, labels):
            if label == -1:
                logger.debug(f"Skipping update for outlier with ID {id_}.")
                continue
            cursor.execute(
                "UPDATE media_face SET cluster_id = %s WHERE id = %s",
                (int(label), int(id_))
            )
        conn.commit()
        logger.info("Cluster IDs successfully written to the database.")


def main():
    load_dotenv()
    logger.info("Environment variables loaded.")
    envs = EnvVars()

    while True:
        try:
            conn = connect_to_database(envs)
        except Exception:
            logger.error("Connection failed. Retrying in 60 seconds.")
            time.sleep(60)
            continue

        rows = get_rows_from_db(conn)
        
        if not rows:
            logger.info("No embeddings found in the database. Waiting for the next cycle.")
            conn.close()
            time.sleep(3600)
            continue

        user_data = {}
        for row in rows:
            user_id = row[3]
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(row)

        for user_id, user_rows in user_data.items():
            logger.info(f"Processing clusters for user ID: {user_id}")
            
            ids, embeddings_raw = zip(*[(row[0], row[1]) for row in user_rows])
            embeddings = np.array([np.fromstring(embedding.strip("[]"), dtype=np.float32, sep=",") for embedding in embeddings_raw])

            new_labels = cluster_embeddings(embeddings)

            consistent_labels = assign_consistent_cluster_ids(user_rows, new_labels, ids, conn, user_id)

            unique_labels = np.unique(consistent_labels)
            for label in unique_labels:
                if label == -1:
                    logger.info(f"User {user_id} - Outliers: {np.sum(np.array(consistent_labels) == label)}")
                else:
                    logger.info(f"User {user_id} - Cluster {label}: Identified {np.sum(np.array(consistent_labels) == label)} faces.")

            write_clusters_to_db(conn, ids, consistent_labels)

        conn.close()
        logger.info("Database connection closed.")
        
        logger.info("Waiting for the next clustering cycle in 60 minutes.")
        time.sleep(60)

if __name__ == "__main__":
    main()

