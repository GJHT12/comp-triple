from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# テキスト形式トリプルを自然な文章に変換するチェーン
def create_naturalize_chain():
    prompt = ChatPromptTemplate.from_template(
        "Using the information below, write a natural and coherent paragraph in English. Do not omit any details.\n\n{triple}"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return (
        {"triple": lambda x: x["triple"]}
        | prompt
        | llm
        | StrOutputParser()
    )

# テキスト形式トリプルファイルを、各トリプルごとに読み取る
def load_triple_blocks(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return [block.strip() for block in content.strip().split("\n\n") if block.strip()]

# 各トリプルを自然な表現に変換
def naturalize_triples(triple_blocks, chain):
    results = []
    for block in triple_blocks:
        result = chain.invoke({"triple": block})
        results.append(result.strip())
    return results

# 結果をファイルに保存。各トリプルは1段落にまとめられる
def save_naturalized_text(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n\n")

def main():
    load_dotenv()

    TRIPLE_TEXT_PATH = "data/triples_as_text.txt"
    NATURALIZED_TRIPLES_PATH = "data/naturalized_triples.txt"

    chain = create_naturalize_chain()
    triple_blocks = load_triple_blocks(TRIPLE_TEXT_PATH)
    results = naturalize_triples(triple_blocks, chain)
    save_naturalized_text(NATURALIZED_TRIPLES_PATH, results)

    print(f"✅ Done. Naturalized text saved to '{NATURALIZED_TRIPLES_PATH}'.")

if __name__ == "__main__":
    main()
