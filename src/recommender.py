"""
recommender.py
--------------
코사인 유사도 기반 음악 추천 모듈.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale


class MusicRecommender:
    """
    GTZAN 30초 특성 CSV를 기반으로 코사인 유사도 추천을 제공합니다.

    Examples
    --------
    >>> rec = MusicRecommender()
    >>> rec.fit('Data/features_30_sec.csv')
    >>> print(rec.recommend('rock.00000.wav', n=5))
    """

    def __init__(self):
        self.sim_df: pd.DataFrame | None = None
        self.labels_df: pd.DataFrame | None = None

    def fit(self, csv_path: str):
        """30초 특성 CSV를 로드하고 유사도 행렬을 계산합니다."""
        df = pd.read_csv(csv_path, index_col="filename")

        self.labels_df = df[["label"]].copy()
        df = df.drop(columns=["length", "label"])

        # 표준화 (평균 0, 표준편차 1)
        df_scaled = pd.DataFrame(
            scale(df), index=df.index, columns=df.columns
        )

        sim = cosine_similarity(df_scaled)
        self.sim_df = pd.DataFrame(sim, index=df_scaled.index, columns=df_scaled.index)
        print(f"유사도 행렬 계산 완료: {self.sim_df.shape} ✅")

    def recommend(self, song_name: str, n: int = 5) -> pd.DataFrame:
        """
        입력 곡과 가장 유사한 n개의 노래를 반환합니다.

        Parameters
        ----------
        song_name : str  파일명 (예: 'rock.00000.wav')
        n         : int  추천 개수

        Returns
        -------
        DataFrame  columns = ['song', 'similarity', 'genre']
        """
        if self.sim_df is None:
            raise RuntimeError("먼저 fit()을 호출하세요.")
        if song_name not in self.sim_df.index:
            available = list(self.sim_df.index[:5])
            raise ValueError(
                f"'{song_name}' 를 찾을 수 없습니다.\n"
                f"예시: {available}"
            )

        series = self.sim_df[song_name].sort_values(ascending=False).drop(song_name)
        top_n = series.head(n).reset_index()
        top_n.columns = ["song", "similarity"]
        top_n["genre"] = top_n["song"].map(
            lambda s: self.labels_df.loc[s, "label"]
            if s in self.labels_df.index else "unknown"
        )
        return top_n

    def similarity_heatmap(self, n_songs: int = 30):
        """유사도 행렬의 일부를 히트맵으로 시각화합니다."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        sub = self.sim_df.iloc[:n_songs, :n_songs]
        plt.figure(figsize=(12, 10))
        sns.heatmap(sub, cmap="coolwarm", xticklabels=True, yticklabels=True, vmin=-1, vmax=1)
        plt.title(f"Cosine Similarity Heatmap (first {n_songs} songs)")
        plt.tight_layout()
        plt.show()
