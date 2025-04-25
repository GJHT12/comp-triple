import os
import json
import pandas as pd
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# トリプルのうち欠損している要素を判定
def detect_missing(triple):
    if triple[0] == "?":
        return "subject"
    elif triple[1] == "?":
        return "predicate"
    elif triple[2] == "?":
        return "object"
    else:
        return None

# retrieverによる検索結果（複数のドキュメント）を文字列として結合
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# 出力用のCSVとJSONファイルを初期化
def initialize_output_files(csv_path, json_path):
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["subject", "predicate", "object"]).to_csv(csv_path, index=False)
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f)

# RAGによるトリプル補完チェーンを作成
def create_rag_chain(retriever):
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an assistant that completes missing parts of a triple. "
            "Respond with only the missing part.\n"
            "Example:\n"
            "Context:\n\n"
            "Triple: ?, is the author of, Tractatus Logico-Philosophicus\n"
            "Ludwig Wittgenstein\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\nComplete the following incomplete triple.\nTriple: {question}"
        ),
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    output_parser = StrOutputParser()

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": lambda _: [],
        }
        | prompt_template
        | llm
        | output_parser
    )

    return rag_chain

# トリプルを補完して返す
def complete_triple(triple, rag_chain):
    missing = detect_missing(triple)
    if not missing:
        return None

    user_input = ", ".join([s if s != "?" else "?" for s in triple])
    response = rag_chain.invoke({
        "question": user_input,
        "chat_history": []
    })

    completed = response.strip().strip('"').strip()
    triple[["subject", "predicate", "object"].index(missing)] = completed
    return user_input, triple

# 補完されたトリプルをCSVとJSONに保存
def save_completed_triple(csv_path, json_path, triple, user_input):
    pd.DataFrame([triple], columns=["subject", "predicate", "object"]).to_csv(csv_path, mode="a", header=False, index=False)
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    json_data.append({"input": user_input, "completed": triple})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

def main():
    load_dotenv()

    # ベクトルDBから検索するretrieverを作成
    retriever = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="db"
    ).as_retriever(search_kwargs={"k": 3})

    rag_chain = create_rag_chain(retriever)

    INCOMPLETE_TRIPLES_CSV = "triples_io/incomplete_triples.csv"
    COMPLETED_TRIPLES_CSV = "triples_io/completed_triples.csv"
    COMPLETED_TRIPLES_JSON = "triples_io/completed_triples.json"

    initialize_output_files(COMPLETED_TRIPLES_CSV, COMPLETED_TRIPLES_JSON)

    input_df = pd.read_csv(INCOMPLETE_TRIPLES_CSV)

    for _, row in input_df.iterrows():
        triple = [str(row["subject"]).strip(), str(row["predicate"]).strip(), str(row["object"]).strip()]
        result = complete_triple(triple, rag_chain)
        if result:
            user_input, completed_triple = result
            save_completed_triple(COMPLETED_TRIPLES_CSV, COMPLETED_TRIPLES_JSON, completed_triple, user_input)

    print(f"✅ Done. Completed triples saved to '{COMPLETED_TRIPLES_CSV}' and '{COMPLETED_TRIPLES_JSON}'.")


if __name__ == "__main__":
    main()
