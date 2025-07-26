import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import pickle

import re
from collections import Counter, defaultdict
import warnings
from scipy import sparse
import logging

from DatabaseManager import DatabaseManager

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class HighConfidenceFeatureExtractor:
    """Yüksek güven skorlu özellik çıkarımı sınıfı"""

    def __init__(self):
        self.author_authority_scores = {}
        self.publisher_quality_scores = {}
        self.category_coherence_scores = {}
        self.genre_compatibility_matrix = {}

    def extract_enhanced_text_features(self, text):
        """Gelişmiş metin özellik çıkarımı"""
        if pd.isna(text) or text == '':
            return self._get_empty_text_features()

        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)

        # Temel özellikler
        basic_features = {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
        }

        # Gelişmiş özellikler
        advanced_features = {
            'readability_score': self._calculate_readability(words, sentences),
            'vocabulary_richness': self._calculate_vocabulary_richness(words),
            'content_density': self._calculate_content_density(text, words),
            'emotional_intensity': self._calculate_emotional_intensity(words),
            'technical_complexity': self._calculate_technical_complexity(words)
        }

        return {**basic_features, **advanced_features}

    def _get_empty_text_features(self):
        """Boş metin için varsayılan özellikler"""
        return {
            'word_count': 0, 'char_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'unique_word_ratio': 0, 'readability_score': 0,
            'vocabulary_richness': 0, 'content_density': 0, 'emotional_intensity': 0,
            'technical_complexity': 0
        }

    def _calculate_readability(self, words, sentences):
        """Okunabilirlik skoru hesapla"""
        if not words or not sentences:
            return 0
        avg_sentence_length = len(words) / max(len([s for s in sentences if s.strip()]), 1)
        avg_word_length = np.mean([len(word) for word in words])
        return min(1.0, (10 - avg_sentence_length / 10 - avg_word_length / 5) / 10)

    def _calculate_vocabulary_richness(self, words):
        """Kelime zenginliği hesapla"""
        if not words:
            return 0
        unique_words = len(set(words))
        total_words = len(words)
        return min(1.0, unique_words / max(total_words, 1))

    def _calculate_content_density(self, text, words):
        """İçerik yoğunluğu hesapla"""
        if not words:
            return 0
        # Noktalama işaretleri ve rakamların oranı
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        number_count = len(re.findall(r'\d+', text))
        return min(1.0, (punctuation_count + number_count) / max(len(text), 1) * 10)

    def _calculate_emotional_intensity(self, words):
        """Duygusal yoğunluk hesapla (basit sözlük tabanlı)"""
        emotional_words = {'love', 'hate', 'fear', 'joy', 'anger', 'sad', 'happy',
                           'excited', 'disappointed', 'amazing', 'terrible', 'wonderful'}
        emotional_count = sum(1 for word in words if word in emotional_words)
        return min(1.0, emotional_count / max(len(words), 1) * 20)

    def _calculate_technical_complexity(self, words):
        """Teknik karmaşıklık hesapla"""
        technical_indicators = {'analysis', 'method', 'system', 'process', 'theory',
                                'research', 'study', 'data', 'result', 'conclusion'}
        technical_count = sum(1 for word in words if word in technical_indicators)
        return min(1.0, technical_count / max(len(words), 1) * 15)

    def calculate_author_authority(self, df):
        """Yazar otoritesi hesapla"""
        author_stats = df.groupby('author').agg({
            'total_ratings': 'sum',
            'average_rating': 'mean',
            'id': 'count',
            'publication_year': ['min', 'max']
        }).reset_index()

        author_stats.columns = ['author', 'total_ratings_sum', 'avg_rating_mean',
                                'book_count', 'first_pub_year', 'last_pub_year']

        # Yazar deneyimi (yıl aralığı)
        author_stats['experience_years'] = author_stats['last_pub_year'] - author_stats['first_pub_year'] + 1

        # Yıllık ortalama kitap sayısı
        author_stats['books_per_year'] = author_stats['book_count'] / np.maximum(author_stats['experience_years'], 1)

        # Kompozit otorite skoru
        scaler = MinMaxScaler()
        features = ['total_ratings_sum', 'avg_rating_mean', 'book_count', 'books_per_year']
        normalized_features = scaler.fit_transform(author_stats[features].fillna(0))

        # Ağırlıklı toplam (rating ve deneyim daha önemli)
        weights = [0.3, 0.4, 0.2, 0.1]
        author_stats['authority_score'] = np.average(normalized_features, axis=1, weights=weights)

        return dict(zip(author_stats['author'], author_stats['authority_score']))

    def calculate_publisher_quality(self, df):
        """Yayınevi kalitesi hesapla"""
        publisher_stats = df.groupby('publisher').agg({
            'average_rating': 'mean',
            'total_ratings': 'sum',
            'id': 'count'
        }).reset_index()

        publisher_stats.columns = ['publisher', 'avg_rating', 'total_ratings', 'book_count']

        # Minimum kitap sayısı filtresi (güvenilirlik için)
        publisher_stats = publisher_stats[publisher_stats['book_count'] >= 3]

        # Kalite skoru hesapla
        scaler = MinMaxScaler()
        features = ['avg_rating', 'total_ratings', 'book_count']
        normalized_features = scaler.fit_transform(publisher_stats[features].fillna(0))

        weights = [0.5, 0.3, 0.2]  # Rating en önemli
        publisher_stats['quality_score'] = np.average(normalized_features, axis=1, weights=weights)

        return dict(zip(publisher_stats['publisher'], publisher_stats['quality_score']))

    def calculate_category_coherence(self, df):
        """Kategori tutarlılığı hesapla"""
        category_stats = df.groupby('category').agg({
            'average_rating': ['mean', 'std'],
            'total_ratings': 'mean',
            'publication_year': ['mean', 'std'],
            'id': 'count'
        }).reset_index()

        category_stats.columns = ['category', 'rating_mean', 'rating_std',
                                  'popularity_mean', 'year_mean', 'year_std', 'book_count']

        # Tutarlılık skoru (düşük std sapma = yüksek tutarlılık)
        category_stats['rating_coherence'] = 1 / (1 + category_stats['rating_std'].fillna(1))
        category_stats['temporal_coherence'] = 1 / (1 + category_stats['year_std'].fillna(10) / 10)

        # Kompozit tutarlılık skoru
        category_stats['coherence_score'] = (
                category_stats['rating_coherence'] * 0.6 +
                category_stats['temporal_coherence'] * 0.4
        )

        return dict(zip(category_stats['category'], category_stats['coherence_score']))

    def build_genre_compatibility_matrix(self, df):
        """Tür uyumluluk matrisi oluştur"""
        # Aynı yazarların yazdığı kategoriler arasındaki ilişkiyi hesapla
        author_categories = df.groupby('author')['category'].apply(list).to_dict()

        categories = df['category'].unique()
        compatibility_matrix = defaultdict(lambda: defaultdict(float))

        for author, cats in author_categories.items():
            if len(cats) > 1:
                for i, cat1 in enumerate(cats):
                    for cat2 in cats[i + 1:]:
                        compatibility_matrix[cat1][cat2] += 1
                        compatibility_matrix[cat2][cat1] += 1

        # Normalize et
        for cat1 in compatibility_matrix:
            total = sum(compatibility_matrix[cat1].values())
            if total > 0:
                for cat2 in compatibility_matrix[cat1]:
                    compatibility_matrix[cat1][cat2] /= total

        return dict(compatibility_matrix)


class UltraHighConfidenceBookRecommendationModel:
    """Ultra yüksek güven skorlu kitap öneri modeli"""

    def __init__(self):
        self.books_df = None
        self.feature_extractors = {}
        self.vectorizers = {}
        self.matrices = {}
        self.dimension_reducers = {}
        self.clusterers = {}
        self.scalers = {}
        self.similarity_models = {}
        self.confidence_models = {}
        self.feature_extractor = HighConfidenceFeatureExtractor()
        self.is_trained = False

        # Gelişmiş hiperparametreler
        self.hyperparameters = {
            'tfidf_max_features': 15000,
            'tfidf_ngram_range': (1, 4),
            'svd_components': 200,
            'nmf_components': 100,
            'pca_components': 150,
            'kmeans_clusters': 30,
            'dbscan_eps': 0.3,
            'knn_neighbors': 100,
            'confidence_threshold': 0.80
        }

    def advanced_hyperparameter_optimization(self, sample_df):
        """İleri seviye hiperparametre optimizasyonu"""
        logger.info("🔧 Gelişmiş hiperparametre optimizasyonu başlıyor...")

        sample_size = min(2000, len(sample_df))
        sample = sample_df.sample(n=sample_size, random_state=42)

        # Grid search parametreleri
        param_combinations = [
            {'max_features': 12000, 'ngram_range': (1, 3), 'svd_components': 150},
            {'max_features': 15000, 'ngram_range': (1, 4), 'svd_components': 200},
            {'max_features': 18000, 'ngram_range': (1, 3), 'svd_components': 250},
        ]

        best_score = 0
        best_params = {}

        for params in param_combinations:
            try:
                score = self._evaluate_parameter_combination(sample, params)
                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Parametre kombinasyonu atlandı: {e}")
                continue

        if best_params:
            self.hyperparameters.update(best_params)
            logger.info(f"✅ Optimal parametreler: {best_params}, Skor: {best_score:.4f}")
        else:
            logger.info("⚠️ Varsayılan parametreler kullanılacak")

    def _evaluate_parameter_combination(self, sample_df, params):
        """Parametre kombinasyonunu değerlendir"""
        combined_features = (
                sample_df['title'].fillna('').astype(str) + ' ' +
                sample_df['author'].fillna('').astype(str) + ' ' +
                sample_df['category'].fillna('').astype(str) + ' ' +
                sample_df['description'].fillna('').astype(str)
        )

        vectorizer = TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=params['ngram_range'],
            stop_words='english',
            min_df=2,
            max_df=0.85
        )

        tfidf_matrix = vectorizer.fit_transform(combined_features)

        # SVD ile boyut indirgeme
        n_components = min(params['svd_components'], tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)

        # Kalite metriği: varyans açıklaması + kümeleme kalitesi
        variance_explained = svd.explained_variance_ratio_.sum()

        # Kümeleme kalitesi
        if reduced_matrix.shape[0] > 50:
            kmeans = KMeans(n_clusters=min(10, reduced_matrix.shape[0] // 10), random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_matrix)
            try:
                silhouette_avg = silhouette_score(reduced_matrix, cluster_labels)
            except:
                silhouette_avg = 0
        else:
            silhouette_avg = 0

        # Kompozit skor
        composite_score = variance_explained * 0.7 + (silhouette_avg + 1) / 2 * 0.3
        return composite_score

    def prepare_ultra_features(self, books_df):
        """Ultra gelişmiş özellik hazırlama"""
        logger.info("🔧 Ultra gelişmiş özellikler hazırlanıyor...")

        # Eksik değerleri doldur
        books_df = books_df.fillna({
            'author': 'Bilinmeyen Yazar',
            'category': 'Genel',
            'publisher': 'Bilinmeyen Yayınevi',
            'description': '',
            'average_rating': 0.0,
            'total_ratings': 0,
            'publication_year': 2000,
            'page_count': 200,
            'language': 'unknown'
        })

        # Temel kombinasyonlu özellikler
        books_df['primary_features'] = (
                books_df['title'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['category'].astype(str)
        )

        books_df['secondary_features'] = (
                books_df['description'].astype(str) + ' ' +
                books_df['publisher'].astype(str)
        )

        books_df['combined_features'] = (
                books_df['primary_features'] + ' ' + books_df['secondary_features']
        )

        # Gelişmiş metin özellikleri
        text_features = books_df['description'].apply(
            self.feature_extractor.extract_enhanced_text_features
        )
        text_df = pd.DataFrame(list(text_features))
        books_df = pd.concat([books_df, text_df], axis=1)

        # Otorite ve kalite skorları
        author_authority = self.feature_extractor.calculate_author_authority(books_df)
        publisher_quality = self.feature_extractor.calculate_publisher_quality(books_df)
        category_coherence = self.feature_extractor.calculate_category_coherence(books_df)

        books_df['author_authority'] = books_df['author'].map(author_authority).fillna(0)
        books_df['publisher_quality'] = books_df['publisher'].map(publisher_quality).fillna(0)
        books_df['category_coherence'] = books_df['category'].map(category_coherence).fillna(0)

        # Tür uyumluluk matrisi
        self.genre_compatibility = self.feature_extractor.build_genre_compatibility_matrix(books_df)

        # Gelişmiş türetilmiş özellikler
        current_year = books_df['publication_year'].max()
        books_df['book_age'] = current_year - books_df['publication_year']
        books_df['age_category'] = pd.cut(
            books_df['book_age'],
            bins=[-1, 2, 5, 10, 20, 100],
            labels=['very_new', 'new', 'recent', 'mature', 'classic']
        )

        # Rating güvenilirlik skoru
        books_df['rating_reliability'] = np.log1p(books_df['total_ratings']) / 10
        books_df['adjusted_rating'] = (
                books_df['average_rating'] * books_df['rating_reliability']
        )

        # Popülerlik segmentasyonu
        books_df['popularity_score'] = (
                np.log1p(books_df['total_ratings']) * 0.7 +
                books_df['average_rating'] / 5 * 0.3
        )

        try:
            books_df['popularity_tier'] = pd.qcut(
                books_df['popularity_score'],
                q=5,
                labels=['niche', 'moderate', 'popular', 'trending', 'blockbuster'],
                duplicates='drop'
            )
        except ValueError as e:
            logger.warning(f"⚠️ pd.qcut hatası: {e}. Manuel aralıklar kullanılıyor...")
            bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
            books_df['popularity_tier'] = pd.cut(
                books_df['popularity_score'],
                bins=bins,
                labels=['niche', 'moderate', 'popular', 'trending', 'blockbuster']
            )

        # Kategori içi ranking
        books_df['category_rating_rank'] = books_df.groupby('category')['average_rating'].rank(pct=True)
        books_df['category_popularity_rank'] = books_df.groupby('category')['total_ratings'].rank(pct=True)

        # Yazarın kategori içi performansı
        author_category_stats = books_df.groupby(['author', 'category']).agg({
            'average_rating': 'mean',
            'total_ratings': 'sum',
            'id': 'count'
        }).reset_index()

        author_category_stats['author_category_performance'] = (
                author_category_stats['average_rating'] * 0.5 +
                np.log1p(author_category_stats['total_ratings']) / 10 * 0.3 +
                author_category_stats['id'] / 10 * 0.2
        )

        author_category_perf = dict(
            zip(
                zip(author_category_stats['author'], author_category_stats['category']),
                author_category_stats['author_category_performance']
            )
        )

        books_df['author_category_performance'] = books_df.apply(
            lambda row: author_category_perf.get((row['author'], row['category']), 0),
            axis=1
        )

        return books_df

    def train_ultra_model(self, books_df):
        """Ultra gelişmiş model eğitimi"""
        logger.info("🚀 Ultra gelişmiş model eğitimi başlıyor...")

        # Hiperparametre optimizasyonu
        self.advanced_hyperparameter_optimization(books_df)

        # Ultra özellik hazırlama
        self.books_df = self.prepare_ultra_features(books_df.copy())
        logger.info(f"📊 Toplam kitap sayısı: {len(self.books_df)}")

        # 1. Çoklu vektörizasyon stratejisi
        self._train_multiple_vectorizers()

        # 2. Çoklu boyut indirgeme
        self._train_dimension_reducers()

        # 3. Gelişmiş sayısal özellik mühendisliği
        self._engineer_numerical_features()

        # 4. Çoklu kümeleme yaklaşımı
        self._train_clustering_models()

        # 5. Hibrit benzerlik modelleri
        self._build_similarity_models()

        # 6. Güven skoru modeli eğitimi
        self._train_confidence_models()

        # 7. Final özellik matrisi oluşturma
        self._build_final_feature_matrix()

        self.is_trained = True
        logger.info("✅ Ultra gelişmiş model eğitimi tamamlandı!")
        logger.info(f"📐 Final özellik matrisi boyutu: {self.final_feature_matrix.shape}")

    def _train_multiple_vectorizers(self):
        """Çoklu vektörizasyon eğitimi"""
        logger.info("🔤 Çoklu vektörizasyon eğitimi...")

        # Primary TF-IDF (yazar, kategori odaklı)
        self.vectorizers['primary_tfidf'] = TfidfVectorizer(
            max_features=self.hyperparameters['tfidf_max_features'] + 5000,
            ngram_range=self.hyperparameters['tfidf_ngram_range'],
            stop_words='english',
            min_df=2,
            max_df=0.75,
            sublinear_tf=True
        )

        author_category_features = (
                self.books_df['author'].astype(str) + ' ' +
                self.books_df['author'].astype(str) + ' ' +
                self.books_df['category'].astype(str) + ' ' +
                self.books_df['category'].astype(str) + ' ' +
                self.books_df['title'].astype(str)
        )

        self.matrices['primary_tfidf'] = self.vectorizers['primary_tfidf'].fit_transform(
            author_category_features
        )



        # Secondary TF-IDF (açıklama odaklı)
        self.vectorizers['secondary_tfidf'] = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.85
        )
        self.matrices['secondary_tfidf'] = self.vectorizers['secondary_tfidf'].fit_transform(
            self.books_df['secondary_features']
        )

        # Count Vectorizer (ek perspektif)
        self.vectorizers['count'] = CountVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.matrices['count'] = self.vectorizers['count'].fit_transform(
            self.books_df['combined_features']
        )

    def _train_dimension_reducers(self):
        """Boyut indirgeme modelleri eğitimi"""
        logger.info("📉 Boyut indirgeme modelleri eğitimi...")

        # SVD modelleri
        self.dimension_reducers['primary_svd'] = TruncatedSVD(
            n_components=min(self.hyperparameters['svd_components'],
                             self.matrices['primary_tfidf'].shape[1] - 1),
            random_state=42
        )
        self.matrices['primary_svd'] = self.dimension_reducers['primary_svd'].fit_transform(
            self.matrices['primary_tfidf']
        )

        self.dimension_reducers['secondary_svd'] = TruncatedSVD(
            n_components=min(100, self.matrices['secondary_tfidf'].shape[1] - 1),
            random_state=42
        )
        self.matrices['secondary_svd'] = self.dimension_reducers['secondary_svd'].fit_transform(
            self.matrices['secondary_tfidf']
        )

        # NMF modeli
        self.dimension_reducers['nmf'] = NMF(
            n_components=min(self.hyperparameters['nmf_components'],
                             self.matrices['primary_tfidf'].shape[1] - 1),
            random_state=42,
            max_iter=200
        )
        self.matrices['nmf'] = self.dimension_reducers['nmf'].fit_transform(
            self.matrices['primary_tfidf']
        )

    def _engineer_numerical_features(self):
        """Sayısal özellik mühendisliği"""
        logger.info("🔢 Gelişmiş sayısal özellik mühendisliği...")

        # Temel sayısal özellikler
        basic_numerical = [
            'publication_year', 'average_rating', 'total_ratings', 'page_count',
            'word_count', 'char_count', 'sentence_count', 'avg_word_length',
            'unique_word_ratio', 'readability_score', 'vocabulary_richness',
            'content_density', 'emotional_intensity', 'technical_complexity',
            'author_authority', 'publisher_quality', 'category_coherence',
            'book_age', 'rating_reliability', 'adjusted_rating', 'popularity_score',
            'category_rating_rank', 'category_popularity_rank', 'author_category_performance'
        ]

        # Robust scaler (outlier'lara karşı dayanıklı)
        self.scalers['robust'] = RobustScaler()
        self.matrices['numerical_robust'] = self.scalers['robust'].fit_transform(
            self.books_df[basic_numerical].fillna(0)
        )

        # Standard scaler
        self.scalers['standard'] = StandardScaler()
        self.matrices['numerical_standard'] = self.scalers['standard'].fit_transform(
            self.books_df[basic_numerical].fillna(0)
        )

        # MinMax scaler
        self.scalers['minmax'] = MinMaxScaler()
        self.matrices['numerical_minmax'] = self.scalers['minmax'].fit_transform(
            self.books_df[basic_numerical].fillna(0)
        )

    def _train_clustering_models(self):
        """Kümeleme modelleri eğitimi"""
        logger.info("🎯 Gelişmiş kümeleme modelleri eğitimi...")

        # K-Means clustering
        self.clusterers['kmeans'] = KMeans(
            n_clusters=self.hyperparameters['kmeans_clusters'],
            random_state=42,
            n_init=20
        )

        # Kombine özellikler üzerinde kümeleme
        combined_features_for_clustering = np.hstack([
            self.matrices['primary_svd'],
            self.matrices['numerical_robust']
        ])

        self.matrices['kmeans_clusters'] = self.clusterers['kmeans'].fit_predict(
            combined_features_for_clustering
        ).reshape(-1, 1)

        # Küme merkezlerine uzaklık
        cluster_distances = self.clusterers['kmeans'].transform(combined_features_for_clustering)
        self.matrices['cluster_distances'] = cluster_distances

    def _build_similarity_models(self):
        """Hibrit benzerlik modelleri oluşturma"""
        logger.info("🎯 Hibrit benzerlik modelleri oluşturuluyor...")

        # Çoklu KNN modelleri

        # KNN modelleri
        self.similarity_models['knn_primary'] = NearestNeighbors(
            n_neighbors=self.hyperparameters['knn_neighbors'],
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_primary'].fit(self.matrices['primary_svd'])

        self.similarity_models['knn_numerical'] = NearestNeighbors(
            n_neighbors=self.hyperparameters['knn_neighbors'],
            metric='euclidean',
            algorithm='auto'
        )
        self.similarity_models['knn_numerical'].fit(self.matrices['numerical_robust'])

        # Hibrit benzerlik için özellik kombinasyonu
        self.matrices['hybrid_features'] = np.hstack([
            self.matrices['primary_svd'] * 0.6,
            self.matrices['secondary_svd'] * 0.15,
            self.matrices['nmf'] * 0.15,
            self.matrices['numerical_robust'] * 0.1
        ])

        self.similarity_models['knn_hybrid'] = NearestNeighbors(
            n_neighbors=self.hyperparameters['knn_neighbors'],
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_hybrid'].fit(self.matrices['hybrid_features'])

    def _train_confidence_models(self):
        """Güven skoru modelleri eğitimi"""
        logger.info("🎯 Güven skoru modelleri eğitimi...")

        # Anomali tespiti için Isolation Forest
        # Önce hybrid_features boyutunu al
        hybrid_features_shape = self.matrices['hybrid_features'].shape[1]

        self.confidence_models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.confidence_models['isolation_forest'].fit(self.matrices['hybrid_features'])

        # Güven skoru için özellik kombinasyonu
        # Boyut tutarlılığını sağlamak için sadece belirli özellikleri kullan
        confidence_features = np.hstack([
            self.matrices['numerical_standard'],
            self.matrices['kmeans_clusters'],
            np.mean(self.matrices['cluster_distances'], axis=1).reshape(-1, 1)
        ])

        # Random Forest regressor güven skoru için
        synthetic_confidence_scores = (
                self.books_df['rating_reliability'] * 0.3 +
                self.books_df['author_authority'] * 0.2 +
                self.books_df['publisher_quality'] * 0.2 +
                self.books_df['category_coherence'] * 0.15 +
                (self.books_df['total_ratings'] / self.books_df['total_ratings'].max()) * 0.15
        )

        self.confidence_models['rf_confidence'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.confidence_models['rf_confidence'].fit(confidence_features, synthetic_confidence_scores)
    def _build_final_feature_matrix(self):
        """Final özellik matrisi oluşturma"""
        logger.info("🏗️ Final özellik matrisi oluşturuluyor...")

        # Tüm özellik matrislerini birleştir
        feature_components = [
            self.matrices['primary_svd'],
            self.matrices['secondary_svd'],
            self.matrices['nmf'],
            self.matrices['numerical_robust'],
            self.matrices['kmeans_clusters'],
            np.mean(self.matrices['cluster_distances'], axis=1).reshape(-1, 1)
        ]

        self.final_feature_matrix = np.hstack(feature_components)

        # Final benzerlik modeli
        self.similarity_models['final'] = NearestNeighbors(
            n_neighbors=min(200, len(self.books_df)),
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['final'].fit(self.final_feature_matrix)

    def get_ultra_high_confidence_recommendations(self, book_title, n_recommendations=10):
        """Ultra yüksek güven skorlu öneriler"""
        if not self.is_trained:
            logger.error("❌ Model henüz eğitilmedi!")
            return []

        # Kitabı bul
        book_matches = self.books_df[
            self.books_df['title'].str.contains(book_title, case=False, na=False)
        ]

        if book_matches.empty:
            logger.warning(f"⚠️ '{book_title}' kitabı bulunamadı!")
            return []

        target_book = book_matches.iloc[0]
        target_idx = target_book.name

        logger.info(f"🎯 Hedef kitap: '{target_book['title']}' - {target_book['author']}")

        # Çoklu benzerlik hesaplama
        recommendations = self._calculate_multi_dimensional_similarity(target_idx, n_recommendations * 3)

        # Güven skoru hesaplama
        recommendations = self._calculate_confidence_scores(recommendations, target_book)

        # Çeşitlilik optimizasyonu
        recommendations = self._optimize_diversity(recommendations, target_book)

        # Final filtreleme ve sıralama
        final_recommendations = self._final_ranking_and_filtering(
            recommendations, target_book, n_recommendations
        )

        return final_recommendations

    def _calculate_multi_dimensional_similarity(self, target_idx, n_candidates):
        """Çok boyutlu benzerlik hesaplama"""
        target_features = self.final_feature_matrix[target_idx].reshape(1, -1)

        # Final model ile benzerlik
        distances, indices = self.similarity_models['final'].kneighbors(
            target_features, n_neighbors=n_candidates + 1
        )

        # Kendisini çıkar
        candidate_indices = indices[0][1:]
        candidate_distances = distances[0][1:]

        recommendations = []
        for idx, distance in zip(candidate_indices, candidate_distances):
            book = self.books_df.iloc[idx]
            similarity_score = 1 - distance  # Cosine distance'ı similarity'ye çevir

            recommendations.append({
                'book_data': book,
                'similarity_score': similarity_score,
                'distance': distance,
                'index': idx
            })

        return recommendations

    def _calculate_confidence_scores(self, recommendations, target_book):
        """Güven skorları hesaplama"""
        for rec in recommendations:
            book_data = rec['book_data']

            # Temel güven faktörleri
            rating_confidence = min(1.0, np.log1p(book_data['total_ratings']) / 10)
            authority_confidence = book_data['author_authority']
            quality_confidence = book_data['publisher_quality']
            coherence_confidence = book_data['category_coherence']

            # Kategori uyumluluğu
            category_compatibility = self._get_category_compatibility(
                target_book['category'], book_data['category']
            )
            author_compatibility = 1.0 if book_data['author'] == target_book['author'] else 0.3

            # Anomali skoru (düşük anomali = yüksek güven)
            # Hybrid features boyutuna uygun hale getir
            hybrid_features = self.matrices['hybrid_features'][rec['index']].reshape(1, -1)
            anomaly_score = self.confidence_models['isolation_forest'].decision_function(
                hybrid_features
            )[0]
            anomaly_confidence = (anomaly_score + 0.5) / 1.0  # Normalize

            # ML güven skoru
            confidence_features = np.hstack([
                self.matrices['numerical_standard'][rec['index']],
                self.matrices['kmeans_clusters'][rec['index']],
                np.mean(self.matrices['cluster_distances'][rec['index']])
            ]).reshape(1, -1)

            ml_confidence = self.confidence_models['rf_confidence'].predict(confidence_features)[0]

            # Kompozit güven skoru
            composite_confidence = (
                    rating_confidence * 0.15 +
                    authority_confidence * 0.25 +
                    quality_confidence * 0.10 +
                    coherence_confidence * 0.10 +
                    category_compatibility * 0.25 +
                    author_compatibility * 0.10 +
                    anomaly_confidence * 0.05
            )

            rec['confidence_score'] = min(1.0, max(0.0, composite_confidence))
            rec['confidence_details'] = {
                'rating_confidence': rating_confidence,
                'authority_confidence': authority_confidence,
                'quality_confidence': quality_confidence,
                'category_compatibility': category_compatibility,
                'anomaly_confidence': anomaly_confidence,
                'ml_confidence': ml_confidence
            }

        return recommendations

    def _get_category_compatibility(self, target_category, candidate_category):
        if target_category == candidate_category:
            return 1.0

        if target_category in self.genre_compatibility:
            compatibility = self.genre_compatibility[target_category].get(candidate_category, 0.05)
            return min(0.8, compatibility * 2)

        return 0.05

    def _optimize_diversity(self, recommendations, target_book):
        """Çeşitlilik optimizasyonu"""
        # Yüksek güven skorlu önerileri al
        high_confidence_recs = [
            rec for rec in recommendations
            if rec['confidence_score'] >= self.hyperparameters['confidence_threshold']
        ]

        if len(high_confidence_recs) < 5:
            # Eşik değerini düşür
            threshold = sorted([rec['confidence_score'] for rec in recommendations], reverse=True)[4]
            high_confidence_recs = [
                rec for rec in recommendations
                if rec['confidence_score'] >= threshold
            ]

        # Çeşitlilik için kategori, yazar ve yayınevi dengesi
        diverse_recommendations = []
        used_authors = {target_book['author']}
        used_categories = set()
        used_publishers = set()

        # Sıralama: güven skoru * benzerlik skoru
        high_confidence_recs.sort(
            key=lambda x: x['confidence_score'] * x['similarity_score'],
            reverse=True
        )

        for rec in high_confidence_recs:
            book_data = rec['book_data']

            author_penalty = 0.2 if book_data['author'] in used_authors else 0.0
            category_bonus = 0.3 if book_data['category'] not in used_categories else 0.0
            publisher_bonus = 0.1 if book_data['publisher'] not in used_publishers else 0.0

            same_author_bonus = 0.2 if book_data['author'] == target_book['author'] else 0.0

            diversity_score = 1.0 - author_penalty + category_bonus + publisher_bonus + same_author_bonus
            rec['diversity_score'] = diversity_score
            rec['final_score'] = (
                    rec['confidence_score'] * 0.4 +
                    rec['similarity_score'] * 0.4 +
                    diversity_score * 0.2
            )

            diverse_recommendations.append(rec)

            # Kullanılan değerleri güncelle
            used_authors.add(book_data['author'])
            used_categories.add(book_data['category'])
            used_publishers.add(book_data['publisher'])

        return diverse_recommendations

    def _final_ranking_and_filtering(self, recommendations, target_book, n_recommendations):
        """Final sıralama ve filtreleme"""
        # Final skor ile sırala
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)

        # Top N öneriler
        final_recommendations = []
        for i, rec in enumerate(recommendations[:n_recommendations]):
            book_data = rec['book_data']

            final_rec = {
                'rank': i + 1,
                'title': book_data['title'],
                'author': book_data['author'],
                'category': book_data['category'],
                'publisher': book_data['publisher'],
                'publication_year': int(book_data['publication_year']),
                'average_rating': float(book_data['average_rating']),
                'total_ratings': int(book_data['total_ratings']),
                'page_count': int(book_data['page_count']) if pd.notna(book_data['page_count']) else None,
                'description': book_data['description'][:200] + "..." if len(str(book_data['description'])) > 200 else
                book_data['description'],

                # Skorlar
                'similarity_score': round(rec['similarity_score'], 4),
                'confidence_score': round(rec['confidence_score'], 4),
                'diversity_score': round(rec['diversity_score'], 4),
                'final_score': round(rec['final_score'], 4),

                # Güven detayları
                'confidence_details': {
                    k: round(v, 4) for k, v in rec['confidence_details'].items()
                },

                # Ek bilgiler
                'recommendation_reasons': self._generate_recommendation_reasons(rec, target_book)
            }

            final_recommendations.append(final_rec)

        return final_recommendations

    def _generate_recommendation_reasons(self, rec, target_book):
        """Öneri nedenlerini oluştur"""
        reasons = []
        book_data = rec['book_data']

        # Benzerlik nedenleri
        if rec['similarity_score'] > 0.8:
            reasons.append("Çok yüksek içerik benzerliği")
        elif rec['similarity_score'] > 0.6:
            reasons.append("Yüksek içerik benzerliği")

        # Kategori uyumu
        if book_data['category'] == target_book['category']:
            reasons.append(f"Aynı kategori: {book_data['category']}")

        # Yazar otoritesi
        if book_data['author_authority'] > 0.7:
            reasons.append("Güvenilir yazar")

        # Yayınevi kalitesi
        if book_data['publisher_quality'] > 0.7:
            reasons.append("Kaliteli yayınevi")

        # Rating kalitesi
        if book_data['average_rating'] > 4.0 and book_data['total_ratings'] > 100:
            reasons.append("Yüksek kullanıcı puanı")

        # Popülerlik
        if book_data['total_ratings'] > 1000:
            reasons.append("Popüler kitap")

        return reasons[:3]  # En fazla 3 neden

    def save_model(self, filepath):
        """Modeli kaydet"""
        try:
            model_data = {
                'books_df': self.books_df,
                'vectorizers': self.vectorizers,
                'matrices': {k: v for k, v in self.matrices.items() if not sparse.issparse(v)},
                'dimension_reducers': self.dimension_reducers,
                'clusterers': self.clusterers,
                'scalers': self.scalers,
                'similarity_models': self.similarity_models,
                'confidence_models': self.confidence_models,
                'hyperparameters': self.hyperparameters,
                'is_trained': self.is_trained,
                'final_feature_matrix': self.final_feature_matrix,
                'genre_compatibility': self.genre_compatibility
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"✅ Model kaydedildi: {filepath}")
        except Exception as e:
            logger.error(f"❌ Model kaydetme hatası: {e}")

    def load_model(self, filepath):
        """Modeli yükle"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            for key, value in model_data.items():
                setattr(self, key, value)

            logger.info(f"✅ Model yüklendi: {filepath}")
        except Exception as e:
            logger.error(f"❌ Model yükleme hatası: {e}")


def main():
    """Ana fonksiyon"""
    # Veritabanı bağlantısı
    db_manager = DatabaseManager()

    try:
        # Veri çekme
        books_df = db_manager.get_books_dataframe()
        if books_df.empty:
            logger.error("❌ Veri çekilemedi!")
            return

        logger.info(f"📚 Toplam {len(books_df)} kitap verisi çekildi")

        # Model oluşturma ve eğitme
        model = UltraHighConfidenceBookRecommendationModel()
        model.train_ultra_model(books_df)

        # Örnek öneriler
        test_books = ['suç ve ceza', 'körlük', 'anna karenina']

        for book_title in test_books:
            print(f"\n{'=' * 80}")
            print(f"🔍 '{book_title}' için öneriler:")
            print('=' * 80)

            recommendations = model.get_ultra_high_confidence_recommendations(
                book_title, n_recommendations=5
            )

            if recommendations:
                for rec in recommendations:
                    print(f"\n{rec['rank']}. {rec['title']}")
                    print(f"   Yazar: {rec['author']}")
                    print(f"   Kategori: {rec['category']}")
                    print(f"   Puan: {rec['average_rating']:.1f} ({rec['total_ratings']} değerlendirme)")
                    print(f"   Final Skor: {rec['final_score']:.4f}")
                    print(f"   Güven: {rec['confidence_score']:.4f}")
                    print(f"   Benzerlik: {rec['similarity_score']:.4f}")
                    print(f"   Nedenler: {', '.join(rec['recommendation_reasons'])}")
            else:
                print("❌ Öneri bulunamadı!")

        # Model kaydetme
        model.save_model('ultra_high_confidence_book_model.pkl')

    except Exception as e:
        logger.error(f"❌ Ana fonksiyon hatası: {e}")

    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()