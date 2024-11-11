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
        cursor.execute("SELECT id, embedding, cluster_id FROM media_face")
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


def assign_consistent_cluster_ids(existing_data, new_labels, ids, conn):
    '''
    Assigns consistent cluster IDs across clustering cycles by checking existing clusters in the database.
    Only new faces (with NULL cluster_id) are assigned new cluster IDs if needed.

    Parameters:
    - existing_data: List of tuples containing current rows in the `media_face` table, each in the format 
      (id, embedding, cluster_id). This includes the existing IDs, embeddings, and cluster assignments from the database.
    - new_labels: List of labels produced by the DBSCAN clustering model for each item in `ids`. The label -1 is 
      typically assigned to items classified as outliers (i.e., not belonging to any cluster).
    - ids: List of unique IDs corresponding to the rows in `media_face`, in the same order as `new_labels`.
    - conn: A psycopg connection object used to execute queries to the database.

    Returns:
    - consistent_labels: A list of cluster IDs that assigns a unique, consistent cluster ID to each item in `ids`.
      The list aligns with `ids`, such that each index corresponds to the respective item in `ids`.

    Procedure:
    1. **Initialize Mapping**:
       - `existing_id_to_cluster`: Creates a dictionary mapping existing `id` values to their current `cluster_id` in the
         database. This lets us look up whether an `id` is already assigned to a specific cluster.
       - `max_existing_cluster_id`: Finds the highest existing cluster ID in the database, which ensures new cluster IDs 
         are assigned consecutively from this maximum.

    2. **Assign Cluster IDs**:
       - For each unique label in `new_labels`:
         - **Skip Outliers** (`new_cluster == -1`): Any items labeled as -1 by DBSCAN are considered outliers and are ignored.
         - **Identify Cluster Members**: Gather all `ids` of items sharing the same DBSCAN label, meaning they belong to the
           same cluster as identified in this cycle.
         - **Check for Existing Cluster IDs**: Look up if any of the cluster members already have an existing `cluster_id` 
           in the database:
             - If one or more members have an assigned `cluster_id`, choose one as the assigned ID for this cluster.
             - If none of the members have an existing `cluster_id`, create a new cluster entry in the `cluster` table and 
               use its ID as the `assigned_cluster_id`.
         - **Record Assignment**: Update `new_cluster_assignments` to map each `id` in the cluster to the `assigned_cluster_id`.

    3. **Construct Final Labels**:
       - `consistent_labels`: Create the final list of cluster IDs using `new_cluster_assignments` for any new assignments
         or fall back to existing values in `existing_id_to_cluster`. For any unassigned IDs, use -1.
       
    4. **Return**: Return the `consistent_labels` list for use in the next steps.
    '''
    logger.info("Assigning consistent cluster IDs across clustering cycles.")
    existing_id_to_cluster = {id_: cluster_id for id_, _, cluster_id in existing_data}

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
                cursor.execute("INSERT INTO cluster DEFAULT VALUES RETURNING id")
                assigned_cluster_id = cursor.fetchone()[0]
                conn.commit()
            logger.debug(f"Assigned new cluster ID {assigned_cluster_id}.")

        for member_id in cluster_member_ids:
            new_cluster_assignments[member_id] = assigned_cluster_id

    consistent_labels = [new_cluster_assignments.get(id_, existing_id_to_cluster.get(id_, -1)) for id_ in ids]
    consistent_labels = [label if label is not None else -1 for label in consistent_labels]
    
    logger.info("Consistent cluster IDs assigned.")
    return consistent_labels


def write_clusters_to_db(conn, ids, labels):
    '''
    Updates the database with the cluster IDs for each row in the `media_face` table based on clustering results.

    Parameters:
    - conn: A psycopg connection object used to execute queries to the database.
    - ids: List of unique identifiers (primary keys) from the `media_face` table corresponding to each item in the `labels`.
    - labels: List of cluster IDs for each item in `ids`, where each index corresponds to the same index in `ids`.
      The label -1 typically signifies an outlier, which should not be assigned any `cluster_id`.

    Returns:
    - None

    Procedure:
    **Update Cluster IDs**:
    - For each `id`, the corresponding `label` is checked:
        - If the `label` is -1, this item is classified as an outlier, and no update to `cluster_id` is performed.
        - If the `label` is a positive integer, an SQL `UPDATE` statement is executed to set the `cluster_id` of the
        `media_face` row with `id` to the value in `label`.
        - This function only updates rows with assigned clusters (ignoring outliers), maintaining a `NULL` `cluster_id`
      for any data points that did not fit into a cluster.
    '''
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

        ids, embeddings_raw = zip(*[(row[0], row[1]) for row in rows])
        embeddings = np.array([np.fromstring(embedding.strip("[]"), dtype=np.float32, sep=",") for embedding in embeddings_raw])

        new_labels = cluster_embeddings(embeddings)

        consistent_labels = assign_consistent_cluster_ids(rows, new_labels, ids, conn)

        unique_labels = np.unique(consistent_labels)
        for label in unique_labels:
            if label == -1:
                logger.info(f"Outliers: {np.sum(np.array(consistent_labels) == label)}")
            else:
                logger.info(f"Cluster {label}: Identified {np.sum(np.array(consistent_labels) == label)} faces.")

        write_clusters_to_db(conn, ids, consistent_labels)

        conn.close()
        logger.info("Database connection closed.")

        logger.info("Waiting for the next clustering cycle in 60 minutes.")
        time.sleep(3600)


if __name__ == "__main__":
    main()











# TODO RUN CLUSTER FOR EACH USER ID IN DATABASE


# SELECT JOIN MEDIA_FACE WITH MEDIA ON MEDIA_ID
# 
# e para cada id de user fazer clustering 