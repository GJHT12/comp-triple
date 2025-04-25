from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# テキストファイル読み込みDocumentのリストに変換
# 読み込みは1段落ごと（1つのトリプルごと）
def load_documents_from_file(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.strip().split("\n\n") if p.strip()]
    return [Document(page_content=p) for p in paragraphs]

# Documentをベクトルに変換してChroma DBに保存
def embed_documents(documents, persist_dir, model_name="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=model_name)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore

def main():
    load_dotenv()

    TRIPLES_FILE = "data/naturalized_triples.txt"
    VECTOR_DB_DIR = "db"

    documents = load_documents_from_file(TRIPLES_FILE)
    embed_documents(documents, VECTOR_DB_DIR)

    print(f"✅ Done. Chroma Vector DB saved to '{VECTOR_DB_DIR}'.")

if __name__ == "__main__":
    main()