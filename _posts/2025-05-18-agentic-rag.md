---
layout: post
title: "Agentic RAG"
---

外部のデータソースから関連するドキュメントを検索し挿入することで、LLMが持っていない知識を拡張する技術をRAG (Retrieval-Augmented Generation: 検索拡張生成) と呼ぶ。今までは単にユーザークエリに対して関連するドキュメントを最初に検索する手法が一般的だったが、最近ではAgentの持つツールとして必要に応じてRAGを呼び出す Agentic RAG という手法をよく聞くようになった。

## RAGについて

RAG (Retrieval-Augmented Generation: 検索拡張生成) は、外部のデータソースの情報を取得しプロンプトに挿入することで、LLMが持っていない知識を使って推論を可能にする。LLMは訓練データ以降の知識を持っていない（いわゆる知識のカットオフ）。社内情報や特定ドメインに特化した情報など、一般的でない情報の知識も当然ながら持たない。このような知識を、LLMの再訓練とパラメータ変更なしに拡張させて回答させる一つの手法としてRAGが利用される。

また、一般的な情報であったにしても、特定の理論や事実ではなく、key-valueのペアが幅広くある情報（例えば本のタイトルと著者など、個々の情報がデータベースでより集まるような情報）だと、パラメータ数が多いモデルでもどうしてもハルシネーションが起きてしまう。実際に、OpenAIで本のレコメンドをしてもらうも、ハルシネーションが起きてしまい、実際には存在しないタイトルと著者の組を出力されてしまうのを、私はわりと経験している。RAGを使うと、実際のデータベースから情報を取得してLLMに明示的にプロンプトとして渡すことで、このようなハルシネーションの抑制にも効果がある。

## Agentic RAG以前のRAG

Agentic RAG以前の普通のRAGは、大まかには以下のような仕組みになっている。

1. 事前にドキュメントをチャンク化し、embeddingモデル (OpenAIの text-embedding-3-{small|large}, text-embedding-ada-002 など) を使って、ベクトルデータベースやローカルのファイルとして保存しておく。
2. ユーザーがシステムに与えたクエリに対して、同じembeddingモデルで埋め込みをして、ユーザークエリをベクトル化する。
3. ユーザークエリと、事前に構築していたベクトルデータベースで、類似度検索をする。類似度の指標としてはcos類似度がよく使われる。類似度が上位N位のドキュメント（正確にはドキュメントのチャンク）の一覧を取得する。
4. 取得したN件のドキュメントに対して、rerankモデルを適用し、ユーザークエリのコンテキストを考慮した上でさらに関連する順にドキュメントを並び替える。その後、上位M位のドキュメントを取得する。
5. 最終的に取得したM件のドキュメントを、ユーザークエリに差し込み、知識を拡張したプロンプトを生成する。それをLLMに渡し、最終的な出力を作成する。

5のプロンプトとしては下記のようなものになる。([1]より引用)

```markdown
We have provided context information below.
---------------------
{ context_str }
---------------------
Given this information, please answer the question: { query_str }
```

上記のバリエーションとして、関連するドキュメントを検索する際により単純なBM25などのアルゴリズムを使用する、もしくはBM25とembeddingモデルの類似度検索を両方使用してハイブリッドで検索する、ドキュメントにグラフ構造を付与して利用するなど、バリエーションは色々ある。

個人的には素朴なcos類似度による検索によるRAGしか使ったことはない。なので、BM25による検索やGraph-RAGについては、私はあまりよくわかっていない。

## Agentic RAG

Agentic RAGでは、LLMのAgentの持つツールの1つとして、情報取得を行うツールを与える。これにより、Agentは推論中に、動的にRAGによる情報取得と知識獲得を行うことができる。これにより、たとえば下記のような振る舞いが可能にある。

- RAGの検索で、ユーザークエリの全文を使用するのではなく、検索に必要な部分のみを抽出し検索する。
- LLMの出力の途中でRAGの検索を行う。これにより、明示的にユーザークエリに与えられていなく、CoT (Chain of Thought) の途中で情報取得の必要性が生じた場合でも、うまくRAGが機能する。
- RAGによるドキュメント取得の精度が悪かったり、期待したドキュメントが取得できなかった場合でも、それに応じて再検索を行うことができる。
- RAGによる情報取得で、さらにRAGをする必要性のある新規の概念が生じた場合でも、再びRAGを行うことができる。

Googleの公開しているホワイトペーパー "Agents Companion" ([2]) でも、Agentic RAGは "A Critical Evolution in Retrieval-Augmented Generation" と紹介されており、従来型のRAGよりも良い結果をもたらすと言われている。

このGoogleのホワイトペーパーだと、従来のRAGと比較したAgentic RAGの優れた点として、下記を挙げられている。

> 
> 
> - Context-Aware Query Expansion: Instead of relying on a single search pass, agents generate multiple query refinements to retrieve more relevant and comprehensive results.
> - Multi-Step Reasoning: Agents decompose complex queries into smaller logical steps, retrieving information sequentially to build structured responses.
> - Adaptive Source Selection: Instead of fetching data from a single vector database, retrieval agents dynamically select the best knowledge sources based on context.
> - Validation and Correction: Evaluator agents cross-check retrieved knowledge for hallucinations and contradictions before integrating it into the final response.

Hugging FaceのMLエンジニアである Aymeric Roucher 氏によるAgentic RAGの記事 ([3]) では、逆に、Agentic RAGと比較した際の従来のRAGが劣っている点として、下記のような説明を与えている。

> 
> 
> - It **performs only one retrieval step**: if the results are bad, the generation in turn will be bad.
> - **Semantic similarity is computed with the *user query* as a reference**, which might be suboptimal: for instance, the user query will often be a question and the document containing the true answer will be in affirmative voice, so its similarity score will be downgraded compared to other source documents in the interrogative form, leading to a risk of missing the relevant information.

## Agentic RAGの実装

 Agentic RAGというキャッチーな名前が与えられているが、行っていることはAgentのツールとして情報検索の処理を渡すというものなので、実装は単にLLM Agentと従来通りの情報検索のツールの実装のみになる。OpenAI Agent SDKを使用したコードを簡単に記載する。

まず必要なライブラリをインストールする。`uv` を使用する。

`pyproject.toml`

```toml
[project]
name = "20250518"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "openai>=1.79.0",
    "openai-agents>=0.0.15",
]

```

RAGをするサンプルドキュメントを作成する。

`docs/sample.txt`

```
株式会社AgentLabsは2018年に設立された人工知能研究機関である。
株式会社AgentLabsの本社はサンフランシスコにあり、創業者にSato Ichiroである。
2023年にAgent-1がリリースされ、推論能力と多言語対応で大きな注目を集めた。
```

OpenAI Agent SDKを使用して、ドキュメント検索をツールとして含むAgentを作成する。

```python
"""
FileSearchTool + gpt-4.1-nano で動く最小 Agentic RAG。
最後に VectorStore と File を削除し、課金を残さない。
"""
import os, time, pathlib
import openai
from agents import Agent, Runner, FileSearchTool, enable_verbose_stdout_logging

enable_verbose_stdout_logging()  # ← 標準出力に詳細ログを出す

openai.api_key = os.getenv("OPENAI_API_KEY")
DOC_PATH = pathlib.Path("docs/sample.txt")

# ---------- 1) ファイルをアップロード ----------
file_obj = openai.files.create(
    file=DOC_PATH.open("rb"),
    purpose="assistants"
)

# ---------- 2) Vector Store を作成 ----------
vs = openai.vector_stores.create(
    name="agentic_rag_sample_vs",
    file_ids=[file_obj.id],
)
vs_id = vs.id

# Vector Store が ready になるまで待機
while True:
    status = openai.vector_stores.retrieve(vs_id).status
    if status in ("completed", "ready", "processed"):
        break
    time.sleep(1)

# ---------- 3) Agent & FileSearchTool ----------
agent = Agent(
    name="AgenticRAGSample",
    model="gpt-4.1-nano",  # 最安モデル [oai_citation:1‡OpenAI](https://openai.com/index/gpt-4-1/)
    tools=[FileSearchTool(vector_store_ids=[vs_id], max_num_results=3)],
    instructions="""
あなたは質問回答アシスタントです。
情報が不足していると判断した場合のみ file_search ツールを呼び出し、その結果を根拠として日本語で簡潔に答えてください。
""",
)

# ---------- 4) 実行 ----------
question = "株式会社AgentLabsの設立年と本社所在地は？"
try:
    result = Runner.run_sync(agent, question)
    print("=== Agent の最終回答 ===")
    print(result.final_output)
finally:
    # ---------- 5) 後片付け（課金対策） ----------
openai.vector_stores.delete(vs_id)        # Vector Store 削除 [oai_citation:5‡Postman API Platform](https://www.postman.com/devrel/openai/request/mvb03oc/delete-vector-store)
    openai.files.delete(file_obj.id)          # アップロードしたファイルも削除
```

実行結果は下記のようになる。

```bash
$ uv run main.py
Creating trace Agent workflow with id trace_3353d0be75f04b09a213dac3ed1b8f45
Setting current trace: trace_3353d0be75f04b09a213dac3ed1b8f45
Creating span <agents.tracing.span_data.AgentSpanData object at 0x104119d10> with id None
Running agent AgenticRAGSample (turn 1)
Creating span <agents.tracing.span_data.ResponseSpanData object at 0x1040a3bd0> with id None
Calling LLM gpt-4.1-nano with input:
[
  {
    "content": "\u682a\u5f0f\u4f1a\u793eAgentLabs\u306e\u8a2d\u7acb\u5e74\u3068\u672c\u793e\u6240\u5728\u5730\u306f\uff1f",
    "role": "user"
  }
]
Tools:
[
  {
    "type": "file_search",
    "vector_store_ids": [
      "vs_6829dc77d1588191b77e6574ba445b27"
    ],
    "max_num_results": 3
  }
]
Stream: False
Tool choice: NOT_GIVEN
Response format: NOT_GIVEN
Previous response id: None

LLM resp:
[
  {
    "id": "fs_6829dc7aace48191847a2d39e4cf678508924bc72d1fcc6c",
    "queries": [
      "\u682a\u5f0f\u4f1a\u793eAgentLabs \u8a2d\u7acb\u5e74",
      "\u682a\u5f0f\u4f1a\u793eAgentLabs \u672c\u793e\u6240\u5728\u5730"
    ],
    "status": "completed",
    "type": "file_search_call",
    "results": null
  },
  {
    "id": "msg_6829dc7b8344819187d2408c769f402f08924bc72d1fcc6c",
    "content": [
      {
        "annotations": [
          {
            "file_id": "file-5ZbU9fXZxbChoSoZ3Cice3",
            "index": 41,
            "type": "file_citation",
            "filename": "sample.txt"
          }
        ],
        "text": "\u682a\u5f0f\u4f1a\u793eAgentLabs\u306f2018\u5e74\u306b\u8a2d\u7acb\u3055\u308c\u3001\u672c\u793e\u306f\u30b5\u30f3\u30d5\u30e9\u30f3\u30b7\u30b9\u30b3\u306b\u3042\u308a\u307e\u3059\u3002",
        "type": "output_text"
      }
    ],
    "role": "assistant",
    "status": "completed",
    "type": "message"
  }
]

Resetting current trace
=== Agent の最終回答 ===
株式会社AgentLabsは2018年に設立され、本社はサンフランシスコにあります。
Exported 1 items
Shutting down trace provider
Shutting down trace processor <agents.tracing.processors.BatchTraceProcessor object at 0x10335c440>
Exported 2 items
```

上記の例ではOpenAI Agent SDKを使用しているのでOpenAI の Vector Storeにデータが保存される。別のVector Storeを使用するには、情報取得をするpython関数を作り、`@function_tool`デコレータでラップすれば良い。下記はChroma DBを使用するツールを作成する例である。

```python
import os, pathlib, textwrap
import chromadb
import openai
from agents import Agent, Runner, function_tool

# ------------ 1. ドキュメント読み込み＆Chroma 構築 ------------
DOC_PATH = pathlib.Path("docs/sample.txt")
text = DOC_PATH.read_text(encoding="utf-8")
chunks = textwrap.wrap(text, width=300)  # 超シンプル分割

client = chromadb.Client()
collection = client.create_collection("local_docs")

# Embedding 関数
def embed(texts):
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in res.data]

collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embed(chunks),
)

# ------------ 2. Retriever tool ------------
@function_tool
def retrieve(query: str, top_k: int = 3) -> str:
    """与えられたクエリに最も近い文書チャンクを返す"""
    embed_query = embed([query])[0]
    results = collection.query(
        query_embeddings=[embed_query],
        n_results=top_k,
    )
    return "\n---\n".join(results["documents"][0])
```

## Agentic RAGのアーキテクチャ

あくまで主観的な意見ではあるが、Agentic RAGを実装する場合は、上記のように単純にtoolとしてRAG検索を与えることで十分なケースが多いのではないかと思う。RAGの情報取得をするツールを直接与えるのではなく、RAGの情報取得から回答までを専門とするAgentを別途作成して、Multi-Agentのアーキテクチャにすることもあるようである。

[4]のサーベイ論文ではSection 5で先行研究のAgentic RAGのアーキテクチャを幅広く紹介していて、もしAgentic RAGのアーキテクチャを精緻化する必要がある場合は、参考になるかもしれない。

---

[1] https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/

[2] https://www.kaggle.com/whitepaper-agent-companion

[3] https://huggingface.co/learn/cookbook/rag_evaluation

[4] https://arxiv.org/abs/2501.09136

[5] https://www.postman.com/devrel/openai/request/mvb03oc/delete-vector-store
