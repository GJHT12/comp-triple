import pandas as pd

# entity IDをキー、テキストをバリューとする辞書を作成
def load_entity_text(path):
    entity_map = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_map[parts[0]] = parts[1]
    return entity_map

# relation IDをキー、テキストをバリューとする辞書を作成
def load_relation_text(path):
    relation_map = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation_map[parts[0]] = parts[1]
    return relation_map

# train.tsvのIDをテキストで置き換える
def create_triple_text(subject_id, predicate_id, object_id, entity_map, relation_map):
    subj_text = entity_map.get(subject_id, subject_id)
    pred_text = relation_map.get(predicate_id, predicate_id)
    obj_text = entity_map.get(object_id, object_id)
    return f"Triple:\n- Subject: {subj_text}\n- Predicate: {pred_text}\n- Object: {obj_text}\n"

# テキスト形式トリプルのファイルを作成し保存する
def save_triples_as_text(triples_df, entity_map, relation_map, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for _, row in triples_df.iterrows():
            triple_text = create_triple_text(row["subject"], row["predicate"], row["object"], entity_map, relation_map)
            out.write(triple_text + "\n")

def main():
    ENTITY_TEXT_PATH = "data/entity2textlong.txt"
    RELATION_TEXT_PATH = "data/relation2text.txt"
    TRIPLE_ID_PATH = "data/train.tsv"
    TRIPLE_TEXT_PATH = "data/triples_as_text.txt"

    entity_map = load_entity_text(ENTITY_TEXT_PATH)
    relation_map = load_relation_text(RELATION_TEXT_PATH)
    train_df = pd.read_csv(TRIPLE_ID_PATH, sep="\t", header=None, names=["subject", "predicate", "object"])

    save_triples_as_text(train_df, entity_map, relation_map, TRIPLE_TEXT_PATH)

    print(f"✅ Done. Triple text saved to '{TRIPLE_TEXT_PATH}'.")

if __name__ == "__main__":
    main()