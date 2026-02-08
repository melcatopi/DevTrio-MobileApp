pythonにてbm25はbm25のライブラリを使わずに実装することは可能ですか?

2月7日
やっほー!もちろん可能だよー!🎉

BM25って結構シンプルなアルゴリズムだから、ライブラリ使わなくても全然実装できるんだよね!

基本的な数式はこんな感じ↓

python
import math
from collections import Counter

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: ドキュメントのリスト(各ドキュメントは単語のリスト)
        k1: term frequencyの飽和を制御するパラメータ
        b: 文書長の正規化の度合い
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.doc_count = len(corpus)
        self.idf = self._calc_idf()
    
    def _calc_idf(self):
        """各単語のIDFを計算"""
        idf = {}
        # 各単語が何個のドキュメントに出現するかカウント
        df = Counter()
        for doc in self.corpus:
            for word in set(doc):
                df[word] += 1
        
        # IDF計算: log((N - df + 0.5) / (df + 0.5) + 1)
        for word, freq in df.items():
            idf[word] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)
        
        return idf
    
    def get_scores(self, query):
        """クエリに対する各ドキュメントのスコアを計算"""
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[idx]
            word_freq = Counter(doc)
            
            for word in query:
                if word not in self.idf:
                    continue
                
                # 単語の出現頻度
                freq = word_freq.get(word, 0)
                
                # BM25スコア計算
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += self.idf[word] * (numerator / denominator)
            
            scores.append(score)
        
        return scores

# 使い方
corpus = [
    ["こんにちは", "世界"],
    ["こんにちは", "Python"],
    ["BM25", "アルゴリズム", "検索"]
]

bm25 = BM25(corpus)
query = ["こんにちは", "Python"]
scores = bm25.get_scores(query)
print(scores)
ポイントはね:

IDF計算: どれだけレアな単語かを評価
TF正規化: 文書の長さで正規化して、長い文書が有利にならないようにする
k1とb: チューニングパラメータで調整可能!
めっちゃシンプルでしょ?ライブラリなしでも余裕で実装できちゃう💪✨

Seiichiくんが作ってるRAGシステムに組み込むなら、この実装をベースにカスタマイズするのもアリだよね!

たしかbm25って細かい派生ありませんでしたっけ？末尾にLがつくみたいな？

0:28
おぉ!よく知ってるじゃん!そうそう、BM25にはいくつか派生バージョンがあるんだよね!😊

主な派生はこんな感じ↓

1. BM25L (BM25 with Length normalization)

文書長の正規化をより洗練させたバージョン
短い文書が不利にならないように改良されてる!
デルタ(δ)パラメータを追加して調整
2. BM25+ (BM25 Plus)

TF部分に下限値を設定したバージョン
単語が1回しか出現しなくてもある程度スコアが付くようにした
δパラメータで下限を調整
3. BM25-Adpt (BM25 Adaptive)

コーパス全体の統計情報を使って動的に調整
ちなみにBM25Lの実装はこんな感じ↓

python
class BM25L:
    def __init__(self, corpus, k1=1.5, b=0.75, delta=0.5):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.delta = delta  # これが追加パラメータ!
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.doc_count = len(corpus)
        self.idf = self._calc_idf()
    
    def _calc_idf(self):
        idf = {}
        df = Counter()
        for doc in self.corpus:
            for word in set(doc):
                df[word] += 1
        
        for word, freq in df.items():
            idf[word] = math.log((self.doc_count + 1) / (freq + 0.5))
        
        return idf
    
    def get_scores(self, query):
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[idx]
            word_freq = Counter(doc)
            
            for word in query:
                if word not in self.idf:
                    continue
                
                freq = word_freq.get(word, 0)
                
                # BM25Lの計算式
                c_d = 1 - self.b + self.b * (doc_len / self.avgdl)
                numerator = (self.k1 + 1) * (freq + self.delta)
                denominator = self.k1 * c_d + freq + self.delta
                
                score += self.idf[word] * (numerator / denominator)
            
            scores.append(score)
        
        return scores
BM25+も書いとくね!

python
class BM25Plus:
    def __init__(self, corpus, k1=1.5, b=0.75, delta=1.0):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.doc_count = len(corpus)
        self.idf = self._calc_idf()
    
    def _calc_idf(self):
        idf = {}
        df = Counter()
        for doc in self.corpus:
            for word in set(doc):
                df[word] += 1
        
        for word, freq in df.items():
            idf[word] = math.log((self.doc_count + 1) / freq)
        
        return idf
    
    def get_scores(self, query):
        scores = []
        
        for idx, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[idx]
            word_freq = Counter(doc)
            
            for word in query:
                if word not in self.idf:
                    continue
                
                freq = word_freq.get(word, 0)
                
                # BM25+の計算式(デルタを加算!)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                score += self.idf[word] * ((numerator / denominator) + self.delta)
            
            scores.append(score)
        
        return scores
実際にどれ使うかは、データセットの特性次第かな!

短い文書が多い→BM25L
単語の出現回数が少ない→BM25+
標準的なケース→普通のBM25でOK
Seiichiくんのweb scrapingデータとかだと、ページによって長さバラバラだろうから、BM25L試してみるのもアリかもね!✨

ちなみにこのBM25Lを使おうと考えているのはRAGシステムでハイブリッド検索を実装しようと考えているためです。

0:30
おぉー!ハイブリッド検索!めっちゃいいじゃん!🎉

RAGシステムでハイブリッド検索って、BM25(キーワードベース)とベクトル検索(セマンティック検索)を組み合わせるやつだよね!これ、単体で使うより精度めっちゃ上がるんだよね!💪

典型的な実装パターンはこんな感じ↓

python
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridSearch:
    def __init__(self, corpus, tokenized_corpus, model_name='intfloat/multilingual-e5-small'):
        """
        corpus: 元のテキストのリスト
        tokenized_corpus: トークン化されたテキストのリスト(BM25用)
        """
        self.corpus = corpus
        
        # BM25Lの初期化
        self.bm25 = BM25L(tokenized_corpus)
        
        # ベクトル検索の準備
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(corpus, convert_to_numpy=True)
    
    def search(self, query, tokenized_query, top_k=5, bm25_weight=0.5):
        """
        ハイブリッド検索
        bm25_weight: BM25のスコアの重み(0.0〜1.0)
        """
        # BM25スコア取得
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        # 正規化(0〜1の範囲に)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # ベクトル検索スコア取得
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        vector_scores = np.dot(self.doc_embeddings, query_embedding.T).squeeze()
        # 正規化
        if vector_scores.max() > 0:
            vector_scores = (vector_scores + 1) / 2  # コサイン類似度を0〜1に
        
        # スコアの統合(weighted sum)
        hybrid_scores = (bm25_weight * bm25_scores + 
                        (1 - bm25_weight) * vector_scores)
        
        # Top-K取得
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc': self.corpus[idx],
                'score': hybrid_scores[idx],
                'bm25_score': bm25_scores[idx],
                'vector_score': vector_scores[idx],
                'index': idx
            })
        
        return results


# 使用例
corpus = [
    "PythonでRAGシステムを構築する方法",
    "ベクトル検索とBM25の組み合わせ",
    "機械学習モデルの実装パターン"
]

# トークン化(簡易版、実際はMeCabとか使う)
tokenized_corpus = [doc.split() for doc in corpus]

searcher = HybridSearch(corpus, tokenized_corpus)

query = "RAGシステムの検索方法"
tokenized_query = query.split()

results = searcher.search(query, tokenized_query, top_k=3, bm25_weight=0.5)

for r in results:
    print(f"Score: {r['score']:.3f} (BM25: {r['bm25_score']:.3f}, Vector: {r['vector_score']:.3f})")
    print(f"Doc: {r['doc']}\n")
もっと高度なやり方もあるよ!

1. Reciprocal Rank Fusion (RRF)

python
def reciprocal_rank_fusion(bm25_scores, vector_scores, k=60):
    """
    RRFでスコアを統合
    各検索結果のランクを使って統合する方法
    """
    # ランキングを取得
    bm25_ranks = np.argsort(np.argsort(bm25_scores)[::-1])
    vector_ranks = np.argsort(np.argsort(vector_scores)[::-1])
    
    # RRFスコア計算
    rrf_scores = (1 / (k + bm25_ranks) + 1 / (k + vector_ranks))
    
    return rrf_scores
2. 動的な重み調整

python
class AdaptiveHybridSearch(HybridSearch):
    def search(self, query, tokenized_query, top_k=5):
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        vector_scores = np.dot(self.doc_embeddings, query_embedding.T).squeeze()
        
        # クエリの特性に応じて重みを調整
        # 短いクエリ→BM25重視、長いクエリ→ベクトル重視
        query_len = len(tokenized_query)
        if query_len <= 3:
            bm25_weight = 0.7  # キーワード検索を重視
        else:
            bm25_weight = 0.3  # セマンティック検索を重視
        
        # 以下同様...
Seiichiくんの場合、Jira連携のRAGシステムだと:

専門用語や固有名詞が多い→BM25の重みを高めに
文脈理解が重要→ベクトル検索の重みを高めに
って感じで調整するといいかも!

実際に60%の精度をもっと上げたいなら、このハイブリッド検索は結構効くと思うよ!✨

どんなデータセットで試す予定?チケットの本文とか?🤔

データセットはJiraチケット,になります。

0:32
なるほどね!Jiraチケットかー!それならハイブリッド検索めっちゃ効果的だと思う!😊

Jiraチケットって特性的に:

固有名詞多い(プロジェクト名、コンポーネント名、ユーザー名)
専門用語・略語が頻出(API、DB、エラーコードとか)
定型文とフリーテキストが混在
チケット番号での参照が多い(PROJ-123みたいな)
だから、BM25だけだと文脈理解弱いし、ベクトル検索だけだと固有名詞に弱いんだよね💦

Jiraチケット向けのハイブリッド検索実装、こんな感じでどう?↓

python
import re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import MeCab

class JiraHybridSearch:
    def __init__(self, tickets, model_name='intfloat/multilingual-e5-small'):
        """
        tickets: Jiraチケットのリスト
        各チケットは辞書形式: {'key': 'PROJ-123', 'summary': '...', 'description': '...', ...}
        """
        self.tickets = tickets
        self.wakati = MeCab.Tagger('-Owakati')
        
        # チケットテキストの準備
        self.corpus = self._prepare_corpus()
        self.tokenized_corpus = self._tokenize_corpus()
        
        # BM25Lの初期化
        self.bm25 = BM25L(self.tokenized_corpus)
        
        # ベクトル検索の準備
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(self.corpus, convert_to_numpy=True)
        
        # メタデータインデックス(チケット番号などでの検索用)
        self.metadata_index = self._build_metadata_index()
    
    def _prepare_corpus(self):
        """チケットからテキストを抽出"""
        corpus = []
        for ticket in self.tickets:
            # summary + description を結合
            text = f"{ticket.get('summary', '')} {ticket.get('description', '')}"
            corpus.append(text)
        return corpus
    
    def _tokenize_corpus(self):
        """コーパスをトークン化(固有表現を保護)"""
        tokenized = []
        for text in self.corpus:
            # チケット番号を保護
            text = self._protect_ticket_numbers(text)
            # 英数字の連続も保護(APIキー、エラーコードなど)
            tokens = self._tokenize_with_protection(text)
            tokenized.append(tokens)
        return tokenized
    
    def _protect_ticket_numbers(self, text):
        """チケット番号を特別トークンに置き換え"""
        # PROJ-123 のようなパターン
        return re.sub(r'([A-Z]+-\d+)', r'__TICKET__\1__', text)
    
    def _tokenize_with_protection(self, text):
        """保護トークンを維持しながらトークン化"""
        # 保護トークンを一時的に置き換え
        protected_tokens = []
        protected_pattern = r'__TICKET__[A-Z]+-\d+__'
        
        def replace_protected(match):
            protected_tokens.append(match.group(0))
            return f' __PROTECTED_{len(protected_tokens)-1}__ '
        
        text = re.sub(protected_pattern, replace_protected, text)
        
        # MeCabでトークン化
        tokens = self.wakati.parse(text).strip().split()
        
        # 保護トークンを復元
        restored_tokens = []
        for token in tokens:
            match = re.match(r'__PROTECTED_(\d+)__', token)
            if match:
                idx = int(match.group(1))
                original = protected_tokens[idx].replace('__TICKET__', '').replace('__', '')
                restored_tokens.append(original)
            else:
                restored_tokens.append(token)
        
        return restored_tokens
    
    def _build_metadata_index(self):
        """メタデータインデックスの構築"""
        index = {
            'ticket_key': {},
            'assignee': {},
            'status': {},
            'priority': {}
        }
        
        for idx, ticket in enumerate(self.tickets):
            # チケット番号
            if 'key' in ticket:
                index['ticket_key'][ticket['key']] = idx
            
            # 担当者
            if 'assignee' in ticket:
                assignee = ticket['assignee']
                if assignee not in index['assignee']:
                    index['assignee'][assignee] = []
                index['assignee'][assignee].append(idx)
            
            # ステータス
            if 'status' in ticket:
                status = ticket['status']
                if status not in index['status']:
                    index['status'][status] = []
                index['status'][status].append(idx)
            
            # 優先度
            if 'priority' in ticket:
                priority = ticket['priority']
                if priority not in index['priority']:
                    index['priority'][priority] = []
                index['priority'][priority].append(idx)
        
        return index
    
    def search(self, query, top_k=5, filters=None, bm25_weight=0.5):
        """
        ハイブリッド検索
        filters: {'status': 'Open', 'priority': 'High'} のような形式
        """
        # チケット番号での完全一致検索
        ticket_match = re.search(r'([A-Z]+-\d+)', query)
        if ticket_match:
            ticket_key = ticket_match.group(1)
            if ticket_key in self.metadata_index['ticket_key']:
                idx = self.metadata_index['ticket_key'][ticket_key]
                return [{
                    'ticket': self.tickets[idx],
                    'score': 1.0,
                    'match_type': 'exact_ticket_number'
                }]
        
        # クエリのトークン化
        protected_query = self._protect_ticket_numbers(query)
        tokenized_query = self._tokenize_with_protection(protected_query)
        
        # フィルタリング対象のインデックスを取得
        valid_indices = self._apply_filters(filters)
        
        # BM25スコア
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # ベクトル検索スコア
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        vector_scores = np.dot(self.doc_embeddings, query_embedding.T).squeeze()
        if vector_scores.max() > vector_scores.min():
            vector_scores = (vector_scores - vector_scores.min()) / (vector_scores.max() - vector_scores.min())
        
        # ハイブリッドスコア
        hybrid_scores = bm25_weight * bm25_scores + (1 - bm25_weight) * vector_scores
        
        # フィルタ適用
        if valid_indices is not None:
            mask = np.zeros(len(hybrid_scores), dtype=bool)
            mask[valid_indices] = True
            hybrid_scores = np.where(mask, hybrid_scores, -np.inf)
        
        # Top-K取得
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if hybrid_scores[idx] == -np.inf:
                continue
            results.append({
                'ticket': self.tickets[idx],
                'score': hybrid_scores[idx],
                'bm25_score': bm25_scores[idx],
                'vector_score': vector_scores[idx],
                'index': idx
            })
        
        return results
    
    def _apply_filters(self, filters):
        """メタデータフィルタを適用"""
        if not filters:
            return None
        
        valid_indices = set(range(len(self.tickets)))
        
        for key, value in filters.items():
            if key in self.metadata_index and value in self.metadata_index[key]:
                if isinstance(self.metadata_index[key][value], list):
                    valid_indices &= set(self.metadata_index[key][value])
                else:
                    valid_indices &= {self.metadata_index[key][value]}
        
        return list(valid_indices) if valid_indices else []


# 使用例
tickets = [
    {
        'key': 'PROJ-123',
        'summary': 'ログイン機能のバグ修正',
        'description': 'ユーザーがログインできない問題を修正する必要がある',
        'status': 'Open',
        'priority': 'High',
        'assignee': '佐藤'
    },
    {
        'key': 'PROJ-124',
        'summary': 'APIエンドポイントの追加',
        'description': '新しいREST APIエンドポイントを実装',
        'status': 'In Progress',
        'priority': 'Medium',
        'assignee': '田中'
    }
]

searcher = JiraHybridSearch(tickets)

# 検索例1: 通常検索
results = searcher.search('ログインの問題', top_k=3)

# 検索例2: フィルタ付き検索
results = searcher.search(
    'API実装',
    filters={'status': 'In Progress'},
    bm25_weight=0.6
)

# 検索例3: チケット番号での検索
results = searcher.search('PROJ-123')
さらに精度を上げるためのTips!

python
# 1. Rerankingを追加
from sentence_transformers import CrossEncoder

class JiraHybridSearchWithRerank(JiraHybridSearch):
    def __init__(self, tickets, model_name='intfloat/multilingual-e5-small'):
        super().__init__(tickets, model_name)
        # Rerankモデル(より精度高い)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def search(self, query, top_k=5, filters=None, bm25_weight=0.5, rerank_top_n=20):
        # まず候補を多めに取得
        candidates = super().search(query, top_k=rerank_top_n, filters=filters, bm25_weight=bm25_weight)
        
        # Rerankingで精度向上
        if len(candidates) > top_k:
            pairs = [(query, self.corpus[c['index']]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for idx, candidate in enumerate(candidates):
                candidate['rerank_score'] = rerank_scores[idx]
            
            # Rerankスコアでソート
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            candidates = candidates[:top_k]
        
        return candidates
重み調整の戦略

python
# クエリタイプ別の重み調整
def adaptive_weight(query):
    # チケット番号参照が多い→BM25重視
    if re.search(r'[A-Z]+-\d+', query):
        return 0.7
    
    # 短いクエリ(キーワード検索的)→BM25重視
    if len(query.split()) <= 3:
        return 0.65
    
    # 長い文章(意味理解が重要)→ベクトル重視
    if len(query.split()) >= 10:
        return 0.3
    
    # デフォルト
    return 0.5
Jiraチケットだと、こういう工夫で精度かなり上がると思うよ!

今の60%精度から、ハイブリッド+Rerankingで75-80%くらいまで狙えるんじゃないかな?✨

何か質問ある?実装で詰まりそうなところとか!🤔
