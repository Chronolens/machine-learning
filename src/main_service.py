import threading
import logging
import asyncio
import image_process_service
import cluster_embeddings
import clip_text_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_service")

def run_face_process_service():
    try:
        logger.info("Starting face_process_service...")
        asyncio.run(image_process_service.main())
    except Exception as e:
        logger.error(f"Error in face_process_service: {e}", exc_info=True)

def run_cluster_embeddings():
    try:
        logger.info("Starting cluster_embeddings...")
        cluster_embeddings.main()
    except Exception as e:
        logger.error(f"Error in cluster_embeddings: {e}", exc_info=True)

def run_clip_text_service():
    try:
        logger.info("Starting clip_text_service...")
        asyncio.run(clip_text_process.main())
    except Exception as e:
        logger.error(f"Error in clip_text_service: {e}", exc_info=True)


def main():
    face_process_thread = threading.Thread(target=run_face_process_service, name="FaceProcessServiceThread")
    cluster_embeddings_thread = threading.Thread(target=run_cluster_embeddings, name="ClusterEmbeddingsThread")
    clip_text_thread = threading.Thread(target=run_clip_text_service, name="ClipTextServiceThread")

    face_process_thread.start()
    cluster_embeddings_thread.start()
    clip_text_thread.start()

    face_process_thread.join()
    cluster_embeddings_thread.join()
    clip_text_thread.join()
    logger.info("Both services have terminated.")

if __name__ == "__main__":
    main()
