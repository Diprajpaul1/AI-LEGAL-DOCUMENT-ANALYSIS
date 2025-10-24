import json
import os
from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_ipc_data(file_path: str) -> list[dict]:
    """
    Load IPC data from a JSON file.

    Args:
        file_path (str): Path to the IPC JSON file.

    Returns:
        list[dict]: List of IPC sections as dictionaries.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ IPC JSON file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_documents(ipc_data: list[dict]) -> list[Document]:
    """
    Convert IPC JSON entries to LangChain Document objects.

    Args:
        ipc_data (list[dict]): IPC data loaded from JSON.

    Returns:
        list[Document]: LangChain-compatible documents.
    """
    return [
        Document(
            page_content=f"Section {entry['Section']}: {entry['section_title']}\n\n{entry['section_desc']}",
            metadata={
                "chapter": entry.get("chapter", ""),
                "chapter_title": entry.get("chapter_title", ""),
                "section": entry.get("Section", ""),
                "section_title": entry.get("section_title", "")
            }
        )
        for entry in ipc_data
    ]


def build_ipc_vectordb():
    """
    Build and persist a Chroma vectorstore for IPC sections.
    """

    # ✅ Explicit path to .env file
    env_path = r"C:\Users\dipra\Downloads\AI LEGAL DOC. ANALYSIS\ai-legal-assistant-crewai-main\env_template.txt"

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"❌ .env file not found at: {env_path}")

    # ✅ Load environment variables from .env file
    load_dotenv(dotenv_path=env_path)

    # ✅ Fetch environment variables
    ipc_json_path = os.getenv("IPC_JSON_PATH")
    persist_dir_path = os.getenv("PERSIST_DIRECTORY_PATH")
    collection_name = os.getenv("IPC_COLLECTION_NAME")

    # ✅ Print debug info
    print("\n--- Environment Variables ---")
    print("IPC_JSON_PATH:", ipc_json_path)
    print("PERSIST_DIRECTORY_PATH:", persist_dir_path)
    print("IPC_COLLECTION_NAME:", collection_name)
    print("-----------------------------\n")

    # ✅ Validate environment variables
    if not all([ipc_json_path, persist_dir_path, collection_name]):
        raise EnvironmentError("❌ Missing one or more required environment variables.")

    # ✅ Load and process data
    ipc_data = load_ipc_data(ipc_json_path)
    documents = prepare_documents(ipc_data)

    # ✅ Create embeddings and vectorstore
    print("⚙️  Building vectorstore... please wait.")
    embeddings = HuggingFaceEmbeddings()
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir_path,
        collection_name=collection_name
    )

    print(f"✅ Vectorstore successfully created in collection '{collection_name}' at '{persist_dir_path}'")


if __name__ == "__main__":
    build_ipc_vectordb()
