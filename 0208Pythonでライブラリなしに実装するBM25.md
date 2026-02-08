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
