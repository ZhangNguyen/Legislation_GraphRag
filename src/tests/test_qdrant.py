from src.storage.qdrant_store import get_qdrant_client,ensure_collection
from src.app.settings import settings

def main():
    client = get_qdrant_client()
    ensure_collection(client)
    print("Ok. Collection:",settings.qdrant_collection)

if __name__ == "__main__":
    main()