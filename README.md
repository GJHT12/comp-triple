
---

## 使い方
directory_structure.txtも参照いただけると分かりやすいかと思います。
まずはOpenAIのAPIキーを`.env`に記述

```
OPENAI_API_KEY=your_openai_key
```

次に必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

### 1. 一度subject,predicate,object形式のトリプルを構成する

```bash
python3 generate_triples_as_text.py
```

### 2. 各トリプルについてLLMに渡して自然な文章表現にしてもらう

```bash
python3 naturalize_triples_as_text.py
```

### 3. 自然な表現になったテキストをベクトル化してDBに保存する

```bash
python3 embed_triples.py
```

### 4. RAGによって、DBの検索結果を踏まえて欠損トリプルを補完する

```bash
python3 triple_completion.py
```

---