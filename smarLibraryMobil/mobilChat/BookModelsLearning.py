import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import re
from collections import Counter, defaultdict
import warnings
from scipy import sparse
import logging
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from textblob import TextBlob

# Assuming DatabaseManager exists
try:
    from DatabaseManager import DatabaseManager
except ImportError:
    # Mock DatabaseManager if not available
    class DatabaseManager:
        def __init__(self):
            pass

        def get_books_dataframe(self):
            # Return a sample dataframe for testing
            return pd.DataFrame({
                'id': range(100),
                'title': [f'Book {i}' for i in range(100)],
                'author': [f'Author {i % 20}' for i in range(100)],
                'category': [f'Category {i % 10}' for i in range(100)],
                'publisher': [f'Publisher {i % 15}' for i in range(100)],
                'description': [
                    f'This is a description for book {i}. It contains various themes and topics that make it interesting.'
                    for i in range(100)],
                'average_rating': np.random.uniform(3.0, 5.0, 100),
                'total_ratings': np.random.randint(10, 1000, 100),
                'publication_year': np.random.randint(1990, 2024, 100),
                'page_count': np.random.randint(100, 500, 100),
                'language': ['english'] * 100
            })

        def close_connection(self):
            pass

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLTK data downloads
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}")


class OptimizedNLPProcessor:
    """Optimize edilmiş NLP işlemci sınıfı - GÜVENİLİRLİK ODAKLI"""

    def __init__(self):
        try:
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.stemmer = None
            self.lemmatizer = None
            self.stop_words = set(
                ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

        # Türkçe stop words
        turkish_stop_words = {
            'bir', 'bu', 'şu', 've', 'ile', 'için', 'olan', 'olarak', 'ancak',
            'fakat', 'lakin', 'ama', 'veya', 'ya', 'da', 'de', 'ki', 'gibi',
            'kadar', 'daha', 'en', 'çok', 'az', 'hiç', 've', 'ise', 'eğer',
            'her', 'bütün', 'tüm', 'kendi', 'başka', 'diğer', 'böyle', 'şöyle'
        }
        self.stop_words.update(turkish_stop_words)

        # GÜVENİLİR kitap türlerine özgü anahtar kelimeler - GENİŞLETİLDİ
        self.genre_keywords = {
            'fantasy': {
                'english': ['magic', 'wizard', 'dragon', 'sword', 'kingdom', 'quest', 'adventure',
                            'hero', 'spell', 'enchanted', 'mystical', 'legendary', 'prophecy',
                            'sorcerer', 'fairy', 'elf', 'dwarf', 'mythical', 'realm', 'magical',
                            'fantasy', 'epic', 'warrior', 'battle', 'throne', 'castle', 'princess'],
                'turkish': ['büyü', 'büyücü', 'ejder', 'kılıç', 'krallık', 'macera', 'kahraman',
                            'büyülü', 'efsanevi', 'kehanet', 'peri', 'elf', 'cüce', 'alem',
                            'fantastik', 'epik', 'savaşçı', 'savaş', 'taht', 'kale', 'prenses']
            },
            'mystery': {
                'english': ['detective', 'murder', 'crime', 'investigation', 'clue', 'suspect',
                            'mystery', 'police', 'evidence', 'criminal', 'victim', 'case', 'solve',
                            'thriller', 'suspense', 'investigation', 'forensic', 'killer', 'witness'],
                'turkish': ['dedektif', 'cinayet', 'suç', 'araştırma', 'ipucu', 'şüpheli',
                            'gizem', 'polis', 'kanıt', 'suçlu', 'kurban', 'dava', 'çözmek',
                            'gerilim', 'şüphe', 'soruşturma', 'adli', 'katil', 'tanık']
            },
            'romance': {
                'english': ['love', 'heart', 'passion', 'relationship', 'romantic', 'kiss',
                            'marriage', 'wedding', 'dating', 'couple', 'emotion', 'feeling',
                            'romance', 'affection', 'intimate', 'tender', 'beloved', 'sweetheart'],
                'turkish': ['aşk', 'kalp', 'tutku', 'ilişki', 'romantik', 'öpücük',
                            'evlilik', 'düğün', 'çift', 'duygu', 'his', 'romantizm',
                            'sevgi', 'samimi', 'şefkatli', 'sevgili', 'tatlım']
            },
            'scifi': {
                'english': ['space', 'alien', 'future', 'technology', 'robot', 'spacecraft',
                            'planet', 'galaxy', 'science', 'experiment', 'laboratory', 'invention',
                            'artificial', 'intelligence', 'cyborg', 'android', 'futuristic', 'quantum'],
                'turkish': ['uzay', 'yabancı', 'gelecek', 'teknoloji', 'robot', 'uzay aracı',
                            'gezegen', 'galaksi', 'bilim', 'deney', 'laboratuvar', 'icat',
                            'yapay', 'zeka', 'siborg', 'android', 'fütüristik', 'kuantum']
            },
            'historical': {
                'english': ['history', 'historical', 'war', 'battle', 'ancient', 'century',
                            'empire', 'king', 'queen', 'medieval', 'renaissance', 'revolution',
                            'dynasty', 'civilization', 'heritage', 'tradition', 'chronicle'],
                'turkish': ['tarih', 'tarihi', 'savaş', 'muharebe', 'antik', 'yüzyıl',
                            'imparatorluk', 'kral', 'kraliçe', 'ortaçağ', 'rönesans', 'devrim',
                            'hanedan', 'medeniyet', 'miras', 'gelenek', 'kronik']
            },
            'literary': {
                'english': ['literary', 'literature', 'classic', 'philosophy', 'intellectual',
                            'profound', 'contemplative', 'existential', 'metaphor', 'symbolism',
                            'narrative', 'prose', 'poetic', 'artistic', 'cultural', 'society'],
                'turkish': ['edebi', 'edebiyat', 'klasik', 'felsefe', 'entelektüel',
                            'derin', 'düşünceli', 'varoluşsal', 'metafor', 'sembolizm',
                            'anlatı', 'düzyazı', 'şiirsel', 'sanatsal', 'kültürel', 'toplum']
            }
        }

        # GUVENLİLİR duygu analizi için kelime listeleri - GENİŞLETİLDİ
        self.emotion_words = {
            'positive': {
                'happy', 'joy', 'love', 'hope', 'triumph', 'success', 'beautiful',
                'wonderful', 'amazing', 'fantastic', 'excellent', 'great', 'good',
                'brilliant', 'magnificent', 'outstanding', 'superb', 'delightful',
                'inspiring', 'uplifting', 'cheerful', 'optimistic', 'pleasant',
                'mutlu', 'sevinç', 'aşk', 'umut', 'zafer', 'başarı', 'güzel',
                'harika', 'muhteşem', 'fantastik', 'mükemmel', 'büyük', 'iyi',
                'parlak', 'görkemli', 'olağanüstü', 'süperb', 'keyifli',
                'ilham', 'yükseltici', 'neşeli', 'iyimser', 'hoş'
            },
            'negative': {
                'sad', 'fear', 'anger', 'hate', 'death', 'war', 'violence',
                'terrible', 'horrible', 'awful', 'bad', 'worst', 'evil',
                'tragic', 'devastating', 'disturbing', 'frightening', 'grim',
                'dark', 'sinister', 'menacing', 'threatening', 'dangerous',
                'üzgün', 'korku', 'öfke', 'nefret', 'ölüm', 'savaş', 'şiddet',
                'berbat', 'korkunç', 'kötü', 'en kötü', 'şeytan', 'trajik',
                'yıkıcı', 'rahatsız', 'korkutucu', 'kasvetli', 'karanlık',
                'uğursuz', 'tehditkar', 'tehdit', 'tehlikeli'
            },
            'neutral': {
                'story', 'character', 'plot', 'book', 'novel', 'chapter',
                'narrative', 'text', 'writing', 'author', 'reader', 'page',
                'hikaye', 'karakter', 'olay örgüsü', 'kitap', 'roman', 'bölüm',
                'anlatı', 'metin', 'yazı', 'yazar', 'okuyucu', 'sayfa'
            }
        }

    def extract_high_confidence_semantic_features(self, text):
        """YÜKSEK GÜVENİLİRLİK semantic özellik çıkarımı"""
        if pd.isna(text) or text == '':
            return self._get_empty_semantic_features()

        text = str(text).lower()

        # GÜVENİLİR tokenizasyon
        try:
            if self.stemmer and self.lemmatizer:
                tokens = word_tokenize(text)
                sentences = sent_tokenize(text)
            else:
                tokens = re.findall(r'\b\w+\b', text)
                sentences = re.split(r'[.!?]+', text)
        except:
            tokens = re.findall(r'\b\w+\b', text)
            sentences = re.split(r'[.!?]+', text)

        # Kaliteli stop words temizleme
        clean_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # GÜVENILIR stemming ve lemmatization
        if self.stemmer and self.lemmatizer and len(clean_tokens) > 0:
            try:
                stemmed_tokens = [self.stemmer.stem(token) for token in clean_tokens]
                lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in clean_tokens]
            except:
                stemmed_tokens = clean_tokens
                lemmatized_tokens = clean_tokens
        else:
            stemmed_tokens = clean_tokens
            lemmatized_tokens = clean_tokens

        # GÜVENİLİR POS tagging
        try:
            pos_tags = pos_tag(clean_tokens)
            noun_count = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
            verb_count = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
            adj_count = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
            adv_count = sum(1 for word, pos in pos_tags if pos.startswith('RB'))
        except:
            # Basit rule-based POS tagging
            noun_count = sum(1 for token in clean_tokens if token.endswith(('tion', 'sion', 'ness', 'ity')))
            verb_count = sum(1 for token in clean_tokens if token.endswith(('ing', 'ed', 'ate')))
            adj_count = sum(1 for token in clean_tokens if token.endswith(('ful', 'less', 'ous', 'ive')))
            adv_count = sum(1 for token in clean_tokens if token.endswith('ly'))

        # GÜVENİLİR Named Entity Recognition
        try:
            named_entities = ne_chunk(pos_tags)
            entity_count = sum(1 for chunk in named_entities if hasattr(chunk, 'label'))
        except:
            # Basit named entity detection
            entity_count = sum(1 for token in clean_tokens if token[0].isupper())

        # GÜÇLENDIRILMIŞ genre indicators
        genre_scores = self._calculate_enhanced_genre_indicators(clean_tokens, text)

        # GELİŞTİRİLMİŞ sentiment analysis
        sentiment_scores = self._calculate_robust_sentiment(clean_tokens, text)

        # Kalite metrikleri
        lexical_diversity = len(set(clean_tokens)) / max(len(clean_tokens), 1)
        avg_sentence_length = len(tokens) / max(len([s for s in sentences if s.strip()]), 1)
        avg_word_length = np.mean([len(word) for word in clean_tokens]) if clean_tokens else 0

        # Karmaşıklık göstergeleri
        complex_word_ratio = sum(1 for word in clean_tokens if len(word) > 6) / max(len(clean_tokens), 1)
        syllable_complexity = self._estimate_syllable_complexity(clean_tokens)

        # GÜVENİLİR topic coherence
        topic_coherence = self._calculate_enhanced_topic_coherence(clean_tokens, stemmed_tokens)

        # GÜVENİLİR semantic density
        semantic_density = self._calculate_enhanced_semantic_density(clean_tokens, lemmatized_tokens)

        # Metin kalitesi skorları
        text_quality = self._calculate_text_quality_score(text, clean_tokens)

        return {
            'token_count': len(clean_tokens),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'complex_word_ratio': complex_word_ratio,
            'syllable_complexity': syllable_complexity,
            'noun_ratio': noun_count / max(len(clean_tokens), 1),
            'verb_ratio': verb_count / max(len(clean_tokens), 1),
            'adj_ratio': adj_count / max(len(clean_tokens), 1),
            'adv_ratio': adv_count / max(len(clean_tokens), 1),
            'entity_density': entity_count / max(len(clean_tokens), 1),
            'topic_coherence': topic_coherence,
            'semantic_density': semantic_density,
            'text_quality': text_quality,
            **genre_scores,
            **sentiment_scores
        }

    def _get_empty_semantic_features(self):
        """Boş semantic özellikler"""
        return {
            'token_count': 0, 'sentence_count': 0, 'lexical_diversity': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0, 'complex_word_ratio': 0,
            'syllable_complexity': 0, 'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0,
            'adv_ratio': 0, 'entity_density': 0, 'topic_coherence': 0, 'semantic_density': 0,
            'text_quality': 0, 'fantasy_score': 0, 'mystery_score': 0, 'romance_score': 0,
            'scifi_score': 0, 'historical_score': 0, 'literary_score': 0,
            'positive_sentiment': 0, 'negative_sentiment': 0, 'neutral_sentiment': 0,
            'sentiment_polarity': 0, 'sentiment_intensity': 0, 'sentiment_confidence': 0
        }

    def _calculate_enhanced_genre_indicators(self, tokens, full_text):
        """GELİŞTİRİLMİŞ tür belirleyici skorları"""
        genre_scores = {}

        for genre, keywords in self.genre_keywords.items():
            all_keywords = set(keywords['english'] + keywords['turkish'])

            # Exact matches
            exact_matches = sum(1 for token in tokens if token in all_keywords)

            # Partial matches (substring detection)
            partial_matches = 0
            for keyword in all_keywords:
                if len(keyword) > 4:  # Sadece uzun kelimeler için
                    partial_matches += full_text.count(keyword)

            # Context-aware scoring
            context_bonus = 0
            if genre == 'fantasy' and any(word in full_text for word in ['magic', 'büyü', 'wizard', 'büyücü']):
                context_bonus = 0.1
            elif genre == 'mystery' and any(word in full_text for word in ['detective', 'dedektif', 'crime', 'suç']):
                context_bonus = 0.1
            elif genre == 'romance' and any(word in full_text for word in ['love', 'aşk', 'heart', 'kalp']):
                context_bonus = 0.1

            # Normalized score
            total_matches = exact_matches * 2 + partial_matches
            normalized_score = min(1.0, total_matches / max(len(tokens), 1) * 10)

            genre_scores[f'{genre}_score'] = normalized_score + context_bonus

        return genre_scores

    def _calculate_robust_sentiment(self, tokens, full_text):
        """ROBUST duygu analizi"""
        positive_count = sum(1 for token in tokens if token in self.emotion_words['positive'])
        negative_count = sum(1 for token in tokens if token in self.emotion_words['negative'])
        neutral_count = sum(1 for token in tokens if token in self.emotion_words['neutral'])

        total_emotional = positive_count + negative_count + neutral_count

        # TextBlob ile çapraz doğrulama
        sentiment_confidence = 0.5  # Default confidence
        try:
            blob = TextBlob(full_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            sentiment_confidence = min(1.0, textblob_subjectivity + 0.3)
        except:
            textblob_polarity = 0

        if total_emotional == 0:
            return {
                'positive_sentiment': 0,
                'negative_sentiment': 0,
                'neutral_sentiment': 0,
                'sentiment_polarity': textblob_polarity * 0.5,  # Sadece TextBlob kullan
                'sentiment_intensity': 0,
                'sentiment_confidence': sentiment_confidence
            }

        # Rule-based polarity
        rule_polarity = (positive_count - negative_count) / max(total_emotional, 1)

        # Combined polarity (rule-based + TextBlob)
        combined_polarity = (rule_polarity * 0.7 + textblob_polarity * 0.3)

        intensity = total_emotional / max(len(tokens), 1)

        return {
            'positive_sentiment': positive_count / max(len(tokens), 1),
            'negative_sentiment': negative_count / max(len(tokens), 1),
            'neutral_sentiment': neutral_count / max(len(tokens), 1),
            'sentiment_polarity': combined_polarity,
            'sentiment_intensity': intensity,
            'sentiment_confidence': sentiment_confidence
        }

    def _estimate_syllable_complexity(self, tokens):
        """Hece karmaşıklığı tahmini"""
        if not tokens:
            return 0

        def count_syllables(word):
            word = word.lower()
            syllables = 0
            vowels = 'aeıioöuü'
            previous_was_vowel = False

            for char in word:
                if char in vowels:
                    if not previous_was_vowel:
                        syllables += 1
                    previous_was_vowel = True
                else:
                    previous_was_vowel = False

            return max(1, syllables)

        total_syllables = sum(count_syllables(token) for token in tokens)
        avg_syllables = total_syllables / len(tokens)

        return min(1.0, avg_syllables / 3.0)  # Normalize to 0-1

    def _calculate_enhanced_topic_coherence(self, tokens, stemmed_tokens):
        """GELİŞTİRİLMİŞ konu tutarlılığı"""
        if len(tokens) < 5:
            return 0

        # Stemmed token frekansları
        token_freq = Counter(stemmed_tokens)

        # En sık kullanılan kelimelerin tutarlılığı
        top_words = dict(token_freq.most_common(min(15, len(token_freq))))

        if not top_words:
            return 0

        # Dağılım tutarlılığı
        frequencies = list(top_words.values())
        freq_std = np.std(frequencies) if len(frequencies) > 1 else 0
        freq_mean = np.mean(frequencies)

        consistency_score = 1 / (1 + freq_std / max(freq_mean, 1))

        # Coverage score
        coverage_score = sum(top_words.values()) / max(len(tokens), 1)

        return min(1.0, (consistency_score * 0.6 + coverage_score * 0.4) * 1.5)

    def _calculate_enhanced_semantic_density(self, tokens, lemmatized_tokens):
        """GELİŞTİRİLMİŞ semantik yoğunluk"""
        if not tokens:
            return 0

        # Unique lemma ratio
        unique_lemmas = len(set(lemmatized_tokens))
        basic_density = unique_lemmas / max(len(tokens), 1)

        # Content word ratio (excluding function words)
        function_words = {'be', 'have', 'do', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        content_words = [token for token in tokens if token not in function_words and len(token) > 2]
        content_ratio = len(content_words) / max(len(tokens), 1)

        # Combined semantic density
        semantic_density = (basic_density * 0.6 + content_ratio * 0.4)

        return min(1.0, semantic_density * 1.3)

    def _calculate_text_quality_score(self, text, tokens):
        """Metin kalitesi skoru"""
        if not tokens:
            return 0

        # Length appropriateness
        text_length = len(text)
        length_score = min(1.0, text_length / 200)  # 200+ karakter ideal

        # Punctuation usage
        punctuation_count = len(re.findall(r'[.!?,:;]', text))
        punctuation_ratio = punctuation_count / max(len(text), 1) * 100
        punctuation_score = min(1.0, punctuation_ratio / 5)  # ~5% ideal

        # Capital letter usage (proper nouns, sentence starts)
        capital_count = len(re.findall(r'[A-ZÜĞŞIÖÇ]', text))
        capital_ratio = capital_count / max(len(text), 1) * 100
        capital_score = min(1.0, capital_ratio / 10)  # ~10% ideal

        # Vocabulary richness
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)

        # Combined quality score
        quality_score = (
                length_score * 0.3 +
                punctuation_score * 0.2 +
                capital_score * 0.2 +
                unique_ratio * 0.3
        )

        return min(1.0, quality_score)


class HighConfidenceBookRecommendationModel:
    """YÜKSEK GÜVENİLİRLİK Kitap Öneri Modeli - Yazar ve Kategori Odaklı"""

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
        self.nlp_processor = OptimizedNLPProcessor()
        self.is_trained = False
        self.final_feature_matrix = None
        self.genre_compatibility = {}

        # Optimized hyperparameters - YAZAR ve KATEGORİ odaklı
        self.hyperparameters = {
            # Açıklama ağırlıkları azaltıldı
            'description_tfidf_max_features': 20000,  # Azaltıldı
            'description_tfidf_ngram_range': (1, 4),  # Azaltıldı
            'description_svd_components': 250,  # Azaltıldı

            # Yazar ağırlıkları artırıldı
            'author_tfidf_max_features': 18000,  # Artırıldı
            'author_tfidf_ngram_range': (1, 4),  # Artırıldı
            'author_svd_components': 200,  # Artırıldı

            # Kategori ağırlıkları artırıldı
            'category_tfidf_max_features': 15000,  # Artırıldı
            'category_tfidf_ngram_range': (1, 4),  # Artırıldı
            'category_svd_components': 150,  # Artırıldı

            # Genel parametreler
            'combined_tfidf_max_features': 30000,
            'combined_tfidf_ngram_range': (1, 5),
            'svd_components': 300,
            'nmf_components': 120,
            'pca_components': 180,
            'kmeans_clusters': 25,
            'dbscan_eps': 0.25,
            'knn_neighbors': 150,
            'confidence_threshold': 0.75  # Hedef güven skoru
        }

    def train_high_confidence_model(self, books_df):
        """YÜKSEK GÜVENİLİRLİK model eğitimi - Yazar ve Kategori Odaklı"""
        logger.info("🚀 Yüksek güvenilirlik model eğitimi başlıyor - Yazar ve Kategori Odaklı...")

        # Enhanced feature preparation
        self.books_df = self.prepare_enhanced_features(books_df.copy())
        logger.info(f"📊 Toplam kitap sayısı: {len(self.books_df)}")

        # 1. YAZAR ve KATEGORİ odaklı vektörizasyon
        self._train_author_category_focused_vectorizers()

        # 2. Gelişmiş boyut indirgeme
        self._train_enhanced_dimension_reducers()

        # 3. Güvenilir sayısal özellik mühendisliği
        self._engineer_reliable_numerical_features()

        # 4. Yazar-Kategori odaklı kümeleme
        self._train_author_category_clustering()

        # 5. Çoklu güven skorlu benzerlik modelleri
        self._build_confidence_focused_similarity_models()

        # 6. ENHANCED güven skoru modeli
        self._train_enhanced_confidence_models()

        # 7. Final yüksek güven özellik matrisi
        self._build_high_confidence_final_matrix()

        self.is_trained = True
        logger.info("✅ Yüksek güvenilirlik model eğitimi tamamlandı!")
        logger.info(f"📐 Final özellik matrisi boyutu: {self.final_feature_matrix.shape}")

    def prepare_enhanced_features(self, books_df):
        """YÜKSEK GÜVENİLİRLİK özellik hazırlama - Yazar ve Kategori Odaklı"""
        logger.info("🔧 Yüksek güvenilirlik özellikler hazırlanıyor...")

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

        # YAZAR ODAKLI özellik mühendisliği - EN YÜKSEK AĞIRLIK
        books_df['author_enhanced_features'] = (
                books_df['author'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['title'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['description'].astype(str)
        )

        # KATEGORİ ODAKLI özellik mühendisliği - YÜKSEK AĞIRLIK
        books_df['category_enhanced_features'] = (
                books_df['category'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['category'].astype(str) + ' ' +
                books_df['title'].astype(str) + ' ' +
                books_df['author'].astype(str) + ' ' +
                books_df['description'].astype(str)
        )

        # AÇIKLAMA özelleştirmesi - ORTA AĞIRLIK
        books_df['description_focused'] = (
                books_df['description'].astype(str) + ' ' +
                books_df['description'].astype(str) + ' ' +
                books_df['title'].astype(str) + ' ' +
                books_df['category'].astype(str)
        )

        # Ana kombinasyon - YAZAR ve KATEGORİ ÖNCELİKLİ
        books_df['primary_features'] = (
                books_df['author_enhanced_features'] + ' ' +
                books_df['category_enhanced_features'] + ' ' +
                books_df['description_focused']
        )

        # Gelişmiş NLP özellik çıkarımı
        logger.info("🧠 Yüksek güvenilirlik NLP özellik çıkarımı...")
        text_features = books_df['description'].apply(
            self.nlp_processor.extract_high_confidence_semantic_features
        )
        text_df = pd.DataFrame(list(text_features))
        books_df = pd.concat([books_df, text_df], axis=1)

        # GÜÇLENMIŞ otorite ve kalite skorları
        author_authority = self._calculate_enhanced_author_authority(books_df)
        publisher_quality = self._calculate_enhanced_publisher_quality(books_df)
        category_coherence = self._calculate_enhanced_category_coherence(books_df)

        books_df['author_authority'] = books_df['author'].map(author_authority).fillna(0)
        books_df['publisher_quality'] = books_df['publisher'].map(publisher_quality).fillna(0)
        books_df['category_coherence'] = books_df['category'].map(category_coherence).fillna(0)

        # GELIŞMIŞ tür uyumluluk matrisi
        self.genre_compatibility = self._build_enhanced_genre_compatibility(books_df)

        # Advanced derived features
        current_year = books_df['publication_year'].max()
        books_df['book_age'] = current_year - books_df['publication_year']
        books_df['age_category'] = pd.cut(
            books_df['book_age'],
            bins=[-1, 2, 5, 10, 20, 100],
            labels=['very_new', 'new', 'recent', 'mature', 'classic']
        )

        # ENHANCED rating güvenilirlik skoru
        books_df['rating_reliability'] = np.minimum(
            1.0,
            (np.log1p(books_df['total_ratings']) / 8) *
            (1 + books_df['average_rating'] / 5)  # Rating bonus
        )
        books_df['adjusted_rating'] = (
                books_df['average_rating'] * (0.6 + 0.4 * books_df['rating_reliability'])
        )

        # Popülerlik segmentasyonu - GELİŞTİRİLMİŞ
        books_df['popularity_score'] = (
                np.log1p(books_df['total_ratings']) * 0.5 +
                books_df['average_rating'] / 5 * 0.3 +
                books_df['author_authority'] * 0.2
        )

        try:
            books_df['popularity_tier'] = pd.qcut(
                books_df['popularity_score'],
                q=5,
                labels=['niche', 'moderate', 'popular', 'trending', 'blockbuster'],
                duplicates='drop'
            )
        except ValueError:
            bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
            books_df['popularity_tier'] = pd.cut(
                books_df['popularity_score'],
                bins=bins,
                labels=['niche', 'moderate', 'popular', 'trending', 'blockbuster']
            )

        # YAZAR-KATEGORİ özelleşme skoru - GÜÇLENDİRİLDİ
        author_category_stats = books_df.groupby(['author', 'category']).agg({
            'average_rating': ['mean', 'count'],
            'total_ratings': 'sum',
            'id': 'count'
        }).reset_index()

        author_category_stats.columns = ['author', 'category', 'avg_rating', 'rating_count',
                                         'total_ratings', 'book_count']

        author_category_stats['specialization_score'] = (
                author_category_stats['avg_rating'] * 0.35 +
                np.log1p(author_category_stats['total_ratings']) / 10 * 0.35 +
                np.minimum(1.0, author_category_stats['book_count'] / 3) * 0.20 +
                np.minimum(1.0, author_category_stats['rating_count'] / 50) * 0.10
        )

        author_category_spec = dict(
            zip(
                zip(author_category_stats['author'], author_category_stats['category']),
                author_category_stats['specialization_score']
            )
        )

        books_df['author_category_specialization'] = books_df.apply(
            lambda row: author_category_spec.get((row['author'], row['category']), 0),
            axis=1
        )

        # Kategori içi ranking - GELİŞTİRİLMİŞ
        books_df['category_rating_rank'] = books_df.groupby('category')['adjusted_rating'].rank(pct=True)
        books_df['category_popularity_rank'] = books_df.groupby('category')['popularity_score'].rank(pct=True)
        books_df['author_category_rank'] = books_df.groupby(['author', 'category'])['adjusted_rating'].rank(pct=True)

        return books_df

    def _calculate_enhanced_author_authority(self, df):
        """GELIŞMIŞ yazar otoritesi hesaplama - GÜVENİLİRLİK ARTIRMA"""
        author_stats = df.groupby('author').agg({
            'total_ratings': ['sum', 'mean', 'count'],
            'average_rating': ['mean', 'std', 'count'],
            'id': 'count',
            'publication_year': ['min', 'max', 'std'],
            'category': lambda x: len(set(x)),
            'page_count': 'mean'
        }).reset_index()

        author_stats.columns = ['author', 'total_ratings_sum', 'total_ratings_mean', 'total_ratings_count',
                                'avg_rating_mean', 'avg_rating_std', 'avg_rating_count', 'book_count',
                                'first_pub_year', 'last_pub_year', 'pub_year_std', 'category_diversity',
                                'avg_page_count']

        # İyileştirilmiş özellikler - daha cömert hesaplama
        author_stats['experience_years'] = author_stats['last_pub_year'] - author_stats['first_pub_year'] + 1
        author_stats['books_per_year'] = author_stats['book_count'] / np.maximum(author_stats['experience_years'], 1)
        author_stats['rating_consistency'] = 1 / (1 + author_stats['avg_rating_std'].fillna(0.3))  # Daha cömert
        author_stats['versatility'] = np.log1p(author_stats['category_diversity'])
        author_stats['productivity'] = np.log1p(author_stats['book_count'])
        author_stats['reader_reach'] = np.log1p(author_stats['total_ratings_sum'])
        author_stats['quality_consistency'] = author_stats['rating_consistency'] * author_stats['avg_rating_mean']

        # Minimum yazar otorite puanı garantisi - YENİ
        author_stats['base_authority'] = 0.4  # Temel otorite seviyesi

        # Tek kitaplı yazarlar için özel bonus - YENİ
        author_stats.loc[author_stats['book_count'] == 1, 'single_book_bonus'] = (
                author_stats.loc[author_stats['book_count'] == 1, 'avg_rating_mean'] / 5 * 0.3
        )
        author_stats['single_book_bonus'] = author_stats['single_book_bonus'].fillna(0)

        scaler = MinMaxScaler()
        features = ['total_ratings_sum', 'avg_rating_mean', 'book_count', 'books_per_year',
                    'rating_consistency', 'versatility', 'productivity', 'reader_reach', 'quality_consistency']

        # NaN değerleri daha cömert değerlerle doldur
        for feature in features:
            median_val = author_stats[feature].median()
            author_stats[feature] = author_stats[feature].fillna(max(0.5, median_val))  # Daha yüksek minimum

        normalized_features = scaler.fit_transform(author_stats[features])

        # YAZAR otoritesi için ağırlıklar - daha dengeli
        weights = [0.25, 0.25, 0.15, 0.08, 0.12, 0.05, 0.05, 0.03, 0.02]
        authority_scores = np.average(normalized_features, axis=1, weights=weights)

        # Final otorite skoru - minimum garanti ile
        final_authority = np.maximum(
            author_stats['base_authority'],
            authority_scores * 1.5 + author_stats['single_book_bonus']  # %50 artış + bonus
        )

        return dict(zip(author_stats['author'], final_authority))

    def _calculate_enhanced_publisher_quality(self, df):
        """GELIŞMIŞ yayınevi kalitesi hesaplama"""
        publisher_stats = df.groupby('publisher').agg({
            'average_rating': ['mean', 'std', 'count'],
            'total_ratings': ['sum', 'mean'],
            'id': 'count',
            'category': lambda x: len(set(x)),
            'author': lambda x: len(set(x)),
            'publication_year': ['min', 'max']
        }).reset_index()

        publisher_stats.columns = ['publisher', 'avg_rating_mean', 'avg_rating_std', 'rating_count',
                                   'total_ratings_sum', 'total_ratings_mean', 'book_count',
                                   'category_count', 'author_count', 'first_year', 'last_year']

        # En az 2 kitabı olan yayınevleri
        publisher_stats = publisher_stats[publisher_stats['book_count'] >= 2]

        publisher_stats['rating_consistency'] = 1 / (1 + publisher_stats['avg_rating_std'].fillna(0.5))
        publisher_stats['market_presence'] = np.log1p(publisher_stats['total_ratings_sum'])
        publisher_stats['author_diversity'] = np.log1p(publisher_stats['author_count'])
        publisher_stats['longevity'] = publisher_stats['last_year'] - publisher_stats['first_year'] + 1
        publisher_stats['category_expertise'] = publisher_stats['category_count'] / np.maximum(
            publisher_stats['book_count'], 1)

        scaler = MinMaxScaler()
        features = ['avg_rating_mean', 'total_ratings_sum', 'book_count', 'rating_consistency',
                    'market_presence', 'author_diversity', 'longevity']

        for feature in features:
            publisher_stats[feature] = publisher_stats[feature].fillna(publisher_stats[feature].median())

        normalized_features = scaler.fit_transform(publisher_stats[features])

        weights = [0.35, 0.20, 0.15, 0.15, 0.08, 0.04, 0.03]
        publisher_stats['quality_score'] = np.average(normalized_features, axis=1, weights=weights)

        return dict(zip(publisher_stats['publisher'], publisher_stats['quality_score']))

    def _calculate_enhanced_category_coherence(self, df):
        """GELIŞMIŞ kategori tutarlılığı hesaplama"""
        category_stats = df.groupby('category').agg({
            'average_rating': ['mean', 'std', 'count'],
            'total_ratings': ['mean', 'std', 'sum'],
            'publication_year': ['mean', 'std'],
            'author': lambda x: len(set(x)),
            'publisher': lambda x: len(set(x)),
            'id': 'count',
            'page_count': ['mean', 'std']
        }).reset_index()

        category_stats.columns = ['category', 'rating_mean', 'rating_std', 'rating_count',
                                  'popularity_mean', 'popularity_std', 'popularity_sum',
                                  'year_mean', 'year_std', 'author_count', 'publisher_count',
                                  'book_count', 'page_mean', 'page_std']

        category_stats['rating_coherence'] = 1 / (1 + category_stats['rating_std'].fillna(0.5))
        category_stats['popularity_coherence'] = 1 / (1 + category_stats['popularity_std'].fillna(100) / 100)
        category_stats['temporal_coherence'] = 1 / (1 + category_stats['year_std'].fillna(10) / 10)
        category_stats['page_coherence'] = 1 / (1 + category_stats['page_std'].fillna(50) / 50)
        category_stats['maturity'] = np.log1p(category_stats['book_count']) * np.log1p(category_stats['author_count'])
        category_stats['market_size'] = np.log1p(category_stats['popularity_sum'])

        scaler = MinMaxScaler()
        features = ['rating_coherence', 'popularity_coherence', 'temporal_coherence',
                    'page_coherence', 'maturity', 'market_size']

        for feature in features:
            category_stats[feature] = category_stats[feature].fillna(category_stats[feature].median())

        normalized_features = scaler.fit_transform(category_stats[features])

        weights = [0.30, 0.20, 0.15, 0.10, 0.15, 0.10]
        category_stats['coherence_score'] = np.average(normalized_features, axis=1, weights=weights)

        return dict(zip(category_stats['category'], category_stats['coherence_score']))

    def _build_enhanced_genre_compatibility(self, df):
        """GELIŞMIŞ tür uyumluluk matrisi"""
        # Yazar bazlı uyumluluk
        author_categories = df.groupby('author')['category'].apply(list).to_dict()

        # Rating bazlı uyumluluk
        df['rating_group'] = pd.cut(df['average_rating'], bins=5,
                                    labels=['low', 'below_avg', 'avg', 'good', 'excellent'])
        rating_categories = df.groupby('rating_group')['category'].apply(list).to_dict()

        # Publisher bazlı uyumluluk
        publisher_categories = df.groupby('publisher')['category'].apply(list).to_dict()

        categories = df['category'].unique()
        compatibility_matrix = defaultdict(lambda: defaultdict(float))

        # Yazar bazlı uyumluluk (en yüksek ağırlık)
        for author, cats in author_categories.items():
            if len(cats) > 1:
                for i, cat1 in enumerate(cats):
                    for cat2 in cats[i + 1:]:
                        compatibility_matrix[cat1][cat2] += 3.0  # Artırıldı
                        compatibility_matrix[cat2][cat1] += 3.0

        # Rating bazlı uyumluluk
        for rating_group, cats in rating_categories.items():
            if len(cats) > 1:
                for i, cat1 in enumerate(cats):
                    for cat2 in cats[i + 1:]:
                        compatibility_matrix[cat1][cat2] += 1.5
                        compatibility_matrix[cat2][cat1] += 1.5

        # Publisher bazlı uyumluluk
        for publisher, cats in publisher_categories.items():
            if len(cats) > 1:
                for i, cat1 in enumerate(cats):
                    for cat2 in cats[i + 1:]:
                        compatibility_matrix[cat1][cat2] += 1.0
                        compatibility_matrix[cat2][cat1] += 1.0

        # Normalize
        for cat1 in compatibility_matrix:
            total = sum(compatibility_matrix[cat1].values())
            if total > 0:
                for cat2 in compatibility_matrix[cat1]:
                    compatibility_matrix[cat1][cat2] /= total

        return dict(compatibility_matrix)

    def _train_author_category_focused_vectorizers(self):
        """YAZAR ve KATEGORİ odaklı vektörizasyon"""
        logger.info("🔤 Yazar ve Kategori odaklı vektörizasyon eğitimi...")

        # 1. YAZAR odaklı TF-IDF (EN YÜKSEK AĞIRLIK)
        self.vectorizers['author_enhanced_tfidf'] = TfidfVectorizer(
            max_features=self.hyperparameters['author_tfidf_max_features'],
            ngram_range=self.hyperparameters['author_tfidf_ngram_range'],
            stop_words='english',
            min_df=1,
            max_df=0.70,
            sublinear_tf=True,
            analyzer='word',
            token_pattern=r'(?u)\b\w+\b'
        )
        self.matrices['author_enhanced_tfidf'] = self.vectorizers['author_enhanced_tfidf'].fit_transform(
            self.books_df['author_enhanced_features']
        )

        # 2. KATEGORİ odaklı TF-IDF (YÜKSEK AĞIRLIK)
        self.vectorizers['category_enhanced_tfidf'] = TfidfVectorizer(
            max_features=self.hyperparameters['category_tfidf_max_features'],
            ngram_range=self.hyperparameters['category_tfidf_ngram_range'],
            stop_words='english',
            min_df=1,
            max_df=0.75,
            sublinear_tf=True,
            analyzer='word'
        )
        self.matrices['category_enhanced_tfidf'] = self.vectorizers['category_enhanced_tfidf'].fit_transform(
            self.books_df['category_enhanced_features']
        )

        # 3. AÇIKLAMA TF-IDF (ORTA AĞIRLIK)
        self.vectorizers['description_tfidf'] = TfidfVectorizer(
            max_features=self.hyperparameters['description_tfidf_max_features'],
            ngram_range=self.hyperparameters['description_tfidf_ngram_range'],
            stop_words='english',
            min_df=2,
            max_df=0.80,
            sublinear_tf=True
        )
        self.matrices['description_tfidf'] = self.vectorizers['description_tfidf'].fit_transform(
            self.books_df['description_focused']
        )

        # 4. Kombinasyon TF-IDF
        self.vectorizers['combined_tfidf'] = TfidfVectorizer(
            max_features=self.hyperparameters['combined_tfidf_max_features'],
            ngram_range=self.hyperparameters['combined_tfidf_ngram_range'],
            stop_words='english',
            min_df=2,
            max_df=0.75,
            sublinear_tf=True
        )
        self.matrices['combined_tfidf'] = self.vectorizers['combined_tfidf'].fit_transform(
            self.books_df['primary_features']
        )

        logger.info(f"✅ Vektörizasyon tamamlandı:")
        logger.info(f"   - Yazar matrisi: {self.matrices['author_enhanced_tfidf'].shape}")
        logger.info(f"   - Kategori matrisi: {self.matrices['category_enhanced_tfidf'].shape}")
        logger.info(f"   - Açıklama matrisi: {self.matrices['description_tfidf'].shape}")

    def _train_enhanced_dimension_reducers(self):
        """Gelişmiş boyut indirgeme - Yazar ve Kategori odaklı"""
        logger.info("📉 Yazar ve Kategori odaklı boyut indirgeme...")

        # YAZAR SVD - EN ÖNEMLİ
        max_components = min(
            self.hyperparameters['author_svd_components'],
            self.matrices['author_enhanced_tfidf'].shape[1] - 1,
            self.matrices['author_enhanced_tfidf'].shape[0] - 1
        )
        n_components = max(1, max_components) if max_components > 0 else 1

        self.dimension_reducers['author_enhanced_svd'] = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='randomized'
        )
        self.matrices['author_enhanced_svd'] = self.dimension_reducers['author_enhanced_svd'].fit_transform(
            self.matrices['author_enhanced_tfidf']
        )

        # KATEGORİ SVD - YÜKSEK ÖNEMLİ
        max_components = min(
            self.hyperparameters['category_svd_components'],
            self.matrices['category_enhanced_tfidf'].shape[1] - 1,
            self.matrices['category_enhanced_tfidf'].shape[0] - 1
        )
        n_components = max(1, max_components) if max_components > 0 else 1

        self.dimension_reducers['category_enhanced_svd'] = TruncatedSVD(
            n_components=n_components,
            random_state=42,
            algorithm='randomized'
        )
        self.matrices['category_enhanced_svd'] = self.dimension_reducers['category_enhanced_svd'].fit_transform(
            self.matrices['category_enhanced_tfidf']
        )

        # AÇIKLAMA SVD - ORTA ÖNEMLİ
        max_components = min(
            self.hyperparameters['description_svd_components'],
            self.matrices['description_tfidf'].shape[1] - 1,
            self.matrices['description_tfidf'].shape[0] - 1
        )
        n_components = max(1, max_components) if max_components > 0 else 1

        self.dimension_reducers['description_svd'] = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )
        self.matrices['description_svd'] = self.dimension_reducers['description_svd'].fit_transform(
            self.matrices['description_tfidf']
        )

        # Combined SVD
        max_components = min(
            self.hyperparameters['svd_components'],
            self.matrices['combined_tfidf'].shape[1] - 1,
            self.matrices['combined_tfidf'].shape[0] - 1
        )
        n_components = max(1, max_components) if max_components > 0 else 1

        self.dimension_reducers['combined_svd'] = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )
        self.matrices['combined_svd'] = self.dimension_reducers['combined_svd'].fit_transform(
            self.matrices['combined_tfidf']
        )

        # NMF
        max_components = min(
            self.hyperparameters['nmf_components'],
            self.matrices['combined_tfidf'].shape[1] - 1,
            self.matrices['combined_tfidf'].shape[0] - 1
        )
        n_components = max(1, max_components) if max_components > 0 else 1

        self.dimension_reducers['nmf'] = NMF(
            n_components=n_components,
            random_state=42,
            max_iter=500
        )
        self.matrices['nmf'] = self.dimension_reducers['nmf'].fit_transform(
            self.matrices['combined_tfidf']
        )

    def _engineer_reliable_numerical_features(self):
        """Güvenilir sayısal özellik mühendisliği"""
        logger.info("🔢 Güvenilir sayısal özellik mühendisliği...")

        # Önce DataFrame'deki mevcut sütunları kontrol et
        available_columns = set(self.books_df.columns)

        # Temel sayısal özellikler - sadece mevcut olanları kullan
        base_numerical_features = [
            'publication_year', 'average_rating', 'total_ratings', 'page_count'
        ]

        # NLP özellikleri - NLP processor tarafından eklenenler
        nlp_features = [
            'token_count', 'sentence_count', 'lexical_diversity',
            'avg_sentence_length', 'avg_word_length', 'complex_word_ratio',
            'syllable_complexity', 'noun_ratio', 'verb_ratio', 'adj_ratio',
            'adv_ratio', 'entity_density', 'topic_coherence', 'semantic_density',
            'text_quality', 'fantasy_score', 'mystery_score', 'romance_score',
            'scifi_score', 'historical_score', 'literary_score',
            'positive_sentiment', 'negative_sentiment', 'neutral_sentiment',
            'sentiment_polarity', 'sentiment_intensity', 'sentiment_confidence'
        ]

        # Mühendislik özellikleri - prepare_enhanced_features'da eklenenler
        engineered_features = [
            'author_authority', 'publisher_quality', 'category_coherence',
            'book_age', 'rating_reliability', 'adjusted_rating', 'popularity_score',
            'category_rating_rank', 'category_popularity_rank', 'author_category_rank',
            'author_category_specialization'
        ]

        # Sadece mevcut sütunları seç
        numerical_features = []

        # Temel özellikleri kontrol et ve ekle
        for feature in base_numerical_features:
            if feature in available_columns:
                numerical_features.append(feature)
            else:
                logger.warning(f"⚠️ Temel özellik bulunamadı: {feature}")

        # NLP özelliklerini kontrol et ve ekle
        for feature in nlp_features:
            if feature in available_columns:
                numerical_features.append(feature)
            else:
                logger.warning(f"⚠️ NLP özelliği bulunamadı: {feature}")

        # Mühendislik özelliklerini kontrol et ve ekle
        for feature in engineered_features:
            if feature in available_columns:
                numerical_features.append(feature)
            else:
                logger.warning(f"⚠️ Mühendislik özelliği bulunamadı: {feature}")

        # Eksik özellikler için varsayılan değerler oluştur
        missing_basic_features = {
            'word_count': self.books_df['description'].str.len().fillna(0),
            'char_count': self.books_df['description'].str.len().fillna(0),
            'unique_word_ratio': self.books_df.get('lexical_diversity', 0.5),
            'readability_score': self.books_df.get('text_quality', 0.5),
            'vocabulary_richness': self.books_df.get('semantic_density', 0.5),
            'content_density': self.books_df.get('topic_coherence', 0.5),
            'emotional_intensity': self.books_df.get('sentiment_intensity', 0.5),
            'technical_complexity': self.books_df.get('complex_word_ratio', 0.3),
            'genre_indicators': (
                    self.books_df.get('fantasy_score', 0) +
                    self.books_df.get('mystery_score', 0) +
                    self.books_df.get('romance_score', 0) +
                    self.books_df.get('scifi_score', 0) +
                    self.books_df.get('historical_score', 0) +
                    self.books_df.get('literary_score', 0)
            ).fillna(0.1),
            'narrative_style': self.books_df.get('sentiment_polarity', 0).abs().fillna(0.5)
        }

        # Eksik özellikleri DataFrame'e ekle
        for feature_name, feature_values in missing_basic_features.items():
            if feature_name not in available_columns:
                self.books_df[feature_name] = feature_values
                numerical_features.append(feature_name)
                logger.info(f"✅ Oluşturulan özellik eklendi: {feature_name}")

        logger.info(f"📊 Toplam {len(numerical_features)} sayısal özellik kullanılacak")

        # Güvenli DataFrame seçimi - tüm özelliklerin var olduğundan emin ol
        feature_data = self.books_df[numerical_features].copy()

        # NaN değerleri uygun varsayılan değerlerle doldur
        feature_data = feature_data.fillna({
            'publication_year': 2000,
            'average_rating': 3.0,
            'total_ratings': 0,
            'page_count': 200,
            'word_count': 100,
            'char_count': 500
        }).fillna(0)  # Geri kalan tüm NaN'ları 0 ile doldur

        # Robust scaler
        self.scalers['robust'] = RobustScaler()
        self.matrices['numerical_robust'] = self.scalers['robust'].fit_transform(feature_data)

        # Standard scaler
        self.scalers['standard'] = StandardScaler()
        self.matrices['numerical_standard'] = self.scalers['standard'].fit_transform(feature_data)

        # MinMax scaler
        self.scalers['minmax'] = MinMaxScaler()
        self.matrices['numerical_minmax'] = self.scalers['minmax'].fit_transform(feature_data)

        logger.info(f"✅ Sayısal özellik mühendisliği tamamlandı: {feature_data.shape}")
        logger.info(f"   - Robust scaling: {self.matrices['numerical_robust'].shape}")
        logger.info(f"   - Standard scaling: {self.matrices['numerical_standard'].shape}")
        logger.info(f"   - MinMax scaling: {self.matrices['numerical_minmax'].shape}")

    def _train_author_category_clustering(self):
        """Yazar-Kategori odaklı kümeleme"""
        logger.info("🎯 Yazar-Kategori odaklı kümeleme...")

        # YAZAR ve KATEGORİ ağırlıklı özellik kombinasyonu
        clustering_features = np.hstack([
            self.matrices['author_enhanced_svd'] * 2.0,  # EN YÜKSEK AĞIRLIK
            self.matrices['category_enhanced_svd'] * 1.8,  # YÜKSEK AĞIRLIK
            self.matrices['description_svd'] * 0.8,  # ORTA AĞIRLIK
            self.matrices['numerical_robust'] * 0.3  # DÜŞÜK AĞIRLIK
        ])

        self.clusterers['kmeans'] = KMeans(
            n_clusters=self.hyperparameters['kmeans_clusters'],
            random_state=42,
            n_init=25
        )

        self.matrices['kmeans_clusters'] = self.clusterers['kmeans'].fit_predict(
            clustering_features
        ).reshape(-1, 1)

        # Küme merkezlerine uzaklık
        cluster_distances = self.clusterers['kmeans'].transform(clustering_features)
        self.matrices['cluster_distances'] = cluster_distances

        logger.info(f"✅ Kümeleme tamamlandı: {self.hyperparameters['kmeans_clusters']} küme")

    def _build_confidence_focused_similarity_models(self):
        """Güven odaklı benzerlik modelleri - Yazar ve Kategori Öncelikli"""
        logger.info("🎯 Güven odaklı benzerlik modelleri - Yazar ve Kategori Öncelikli...")

        # YAZAR benzerlik modeli (EN ÖNEMLİ)
        self.similarity_models['knn_author_enhanced'] = NearestNeighbors(
            n_neighbors=min(400, len(self.books_df)),
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_author_enhanced'].fit(self.matrices['author_enhanced_svd'])

        # KATEGORİ benzerlik modeli (YÜKSEK ÖNEMLİ)
        self.similarity_models['knn_category_enhanced'] = NearestNeighbors(
            n_neighbors=min(350, len(self.books_df)),
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_category_enhanced'].fit(self.matrices['category_enhanced_svd'])

        # AÇIKLAMA benzerlik modeli (ORTA ÖNEMLİ)
        self.similarity_models['knn_description'] = NearestNeighbors(
            n_neighbors=min(300, len(self.books_df)),
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_description'].fit(self.matrices['description_svd'])

        # Sayısal özellik benzerlik modeli
        self.similarity_models['knn_numerical'] = NearestNeighbors(
            n_neighbors=self.hyperparameters['knn_neighbors'],
            metric='euclidean',
            algorithm='auto'
        )
        self.similarity_models['knn_numerical'].fit(self.matrices['numerical_robust'])

        # YÜKSEK GÜVENİLİRLİK hibrit özellik kombinasyonu - YAZAR ve KATEGORİ ÖNCELİKLİ
        self.matrices['high_confidence_features'] = np.hstack([
            self.matrices['author_enhanced_svd'] * 2.2,  # EN YÜKSEK AĞIRLIK
            self.matrices['category_enhanced_svd'] * 2.0,  # YÜKSEK AĞIRLIK
            self.matrices['description_svd'] * 0.8,  # ORTA AĞIRLIK
            self.matrices['combined_svd'] * 0.3,  # DÜŞÜK AĞIRLIK
            self.matrices['numerical_robust'] * 0.2  # EN DÜŞÜK AĞIRLIK
        ])

        self.similarity_models['knn_high_confidence'] = NearestNeighbors(
            n_neighbors=self.hyperparameters['knn_neighbors'],
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['knn_high_confidence'].fit(self.matrices['high_confidence_features'])

        logger.info("✅ Güven odaklı benzerlik modelleri hazır")

    def _train_enhanced_confidence_models(self):
        """GELIŞMIŞ güven skoru modelleri"""
        logger.info("🎯 Gelişmiş güven skoru modelleri eğitimi...")

        # Anomali tespiti
        self.confidence_models['isolation_forest'] = IsolationForest(
            contamination=0.03,  # Daha sıkı anomali tespiti
            random_state=42,
            n_estimators=150
        )
        self.confidence_models['isolation_forest'].fit(self.matrices['high_confidence_features'])

        # GÜÇLENDIRILMIŞ synthetic confidence scores - YAZAR ve KATEGORİ AĞIRLIKLI
        synthetic_confidence_scores = (
                self.books_df['rating_reliability'] * 0.12 +  # Azaltıldı
                self.books_df['author_authority'] * 0.35 +  # ARTIRILD - EN ÖNEMLİ
                self.books_df['publisher_quality'] * 0.08 +  # Azaltıldı
                self.books_df['category_coherence'] * 0.25 +  # ARTIRILD - ÇOK ÖNEMLİ
                self.books_df['author_category_specialization'] * 0.20  # ARTIRILD - ÖNEMLİ
        )

        # NLP güven faktörleri ekleme - AĞIRLIK AZALTILDI
        nlp_confidence = (
                self.books_df['semantic_density'] * 0.03 +
                self.books_df['topic_coherence'] * 0.03 +
                self.books_df['text_quality'] * 0.02 +
                self.books_df['lexical_diversity'] * 0.02 +
                self.books_df['sentiment_confidence'] * 0.02
        )

        synthetic_confidence_scores += nlp_confidence

        # Tüm kitaplar için tutarlı özellik matrisi oluştur
        all_confidence_features = []
        for i in range(len(self.books_df)):
            features = self._get_confidence_features(i)
            all_confidence_features.append(features)

        confidence_features_matrix = np.array(all_confidence_features)

        # Özellik boyutunu kaydet
        self.confidence_feature_dim = confidence_features_matrix.shape[1]
        logger.info(f"💡 Güven modeli özellik boyutu: {self.confidence_feature_dim}")

        # Random Forest regressor - geliştirilmiş parametreler
        self.confidence_models['rf_confidence'] = RandomForestRegressor(
            n_estimators=300,  # Artırıldı
            max_depth=15,  # Artırıldı
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt'
        )
        self.confidence_models['rf_confidence'].fit(confidence_features_matrix, synthetic_confidence_scores)

        # Feature importance analizi
        feature_importance = self.confidence_models['rf_confidence'].feature_importances_
        logger.info(f"💡 En önemli güven faktörleri tespit edildi (ortalama önem: {np.mean(feature_importance):.3f})")

    def _get_confidence_features(self, book_idx):
        """Güven skoru hesaplamak için özellik vektörü oluşturur"""
        book_data = self.books_df.iloc[book_idx]

        # Temel özellikler
        features = [
            book_data['author_authority'],
            book_data['category_coherence'],
            book_data['author_category_specialization'],
            book_data['rating_reliability'],
            book_data['publisher_quality'],
            book_data.get('semantic_density', 0),
            book_data.get('topic_coherence', 0),
            book_data.get('text_quality', 0),
            book_data.get('lexical_diversity', 0),
            book_data.get('sentiment_confidence', 0),
            book_data['average_rating'],
            np.log1p(book_data['total_ratings']),
            book_data['popularity_score'],
            book_data['adjusted_rating'],
            book_data['category_rating_rank'],
            book_data['category_popularity_rank'],
            book_data['author_category_rank']
        ]

        # Küme bilgileri
        if 'kmeans_clusters' in self.matrices:
            cluster_id = self.matrices['kmeans_clusters'][book_idx][0]
            features.append(cluster_id)

            # Küme merkezine uzaklık
            if 'cluster_distances' in self.matrices:
                cluster_dist = self.matrices['cluster_distances'][book_idx][cluster_id]
                features.append(cluster_dist)

        # Anomali skoru
        if 'high_confidence_features' in self.matrices and 'isolation_forest' in self.confidence_models:
            anomaly_score = self.confidence_models['isolation_forest'].decision_function(
                self.matrices['high_confidence_features'][book_idx].reshape(1, -1)
            )[0]
            features.append(anomaly_score)

        # Boyut indirgeme sonuçları
        for matrix_name in ['author_enhanced_svd', 'category_enhanced_svd', 'description_svd']:
            if matrix_name in self.matrices:
                features.extend(self.matrices[matrix_name][book_idx][:5])  # İlk 5 bileşen

        return np.array(features, dtype=np.float32)

    def _train_enhanced_confidence_models(self):
        """GELIŞMIŞ güven skoru modelleri"""
        logger.info("🎯 Gelişmiş güven skoru modelleri eğitimi...")

        # Anomali tespiti
        self.confidence_models['isolation_forest'] = IsolationForest(
            contamination=0.03,  # Daha sıkı anomali tespiti
            random_state=42,
            n_estimators=150
        )
        self.confidence_models['isolation_forest'].fit(self.matrices['high_confidence_features'])

        # GÜÇLENDIRILMIŞ synthetic confidence scores - YAZAR ve KATEGORİ AĞIRLIKLI
        synthetic_confidence_scores = (
                self.books_df['rating_reliability'] * 0.12 +  # Azaltıldı
                self.books_df['author_authority'] * 0.35 +  # ARTIRILD - EN ÖNEMLİ
                self.books_df['publisher_quality'] * 0.08 +  # Azaltıldı
                self.books_df['category_coherence'] * 0.25 +  # ARTIRILD - ÇOK ÖNEMLİ
                self.books_df['author_category_specialization'] * 0.20  # ARTIRILD - ÖNEMLİ
        )

        # NLP güven faktörleri ekleme - AĞIRLIK AZALTILDI
        nlp_confidence = (
                self.books_df['semantic_density'] * 0.03 +
                self.books_df['topic_coherence'] * 0.03 +
                self.books_df['text_quality'] * 0.02 +
                self.books_df['lexical_diversity'] * 0.02 +
                self.books_df['sentiment_confidence'] * 0.02
        )

        synthetic_confidence_scores += nlp_confidence

        # Tüm kitaplar için tutarlı özellik matrisi oluştur
        all_confidence_features = []
        for i in range(len(self.books_df)):
            features = self._get_confidence_features(i)
            all_confidence_features.append(features)

        confidence_features_matrix = np.array(all_confidence_features)

        # Özellik boyutunu kaydet
        self.confidence_feature_dim = confidence_features_matrix.shape[1]
        logger.info(f"💡 Güven modeli özellik boyutu: {self.confidence_feature_dim}")

        # Random Forest regressor - geliştirilmiş parametreler
        self.confidence_models['rf_confidence'] = RandomForestRegressor(
            n_estimators=300,  # Artırıldı
            max_depth=15,  # Artırıldı
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt'
        )
        self.confidence_models['rf_confidence'].fit(confidence_features_matrix, synthetic_confidence_scores)

        # Feature importance analizi
        feature_importance = self.confidence_models['rf_confidence'].feature_importances_
        logger.info(f"💡 En önemli güven faktörleri tespit edildi (ortalama önem: {np.mean(feature_importance):.3f})")

    def _build_high_confidence_final_matrix(self):
        """Yüksek güven final özellik matrisi - Yazar ve Kategori Öncelikli"""
        logger.info("🏗️ Yüksek güven final özellik matrisi - Yazar ve Kategori Öncelikli...")

        # YÜKSEK GÜVENİLİRLİK final özellik kombinasyonu - YAZAR ve KATEGORİ MAX AĞIRLIK
        feature_components = [
            self.matrices['author_enhanced_svd'] * 3.0,  # EN YÜKSEK AĞIRLIK - YAZAR
            self.matrices['category_enhanced_svd'] * 2.8,  # ÇOK YÜKSEK AĞIRLIK - KATEGORİ
            self.matrices['description_svd'] * 1.0,  # ORTA AĞIRLIK - AÇIKLAMA
            self.matrices['combined_svd'] * 0.4,  # DÜŞÜK AĞIRLIK
            self.matrices['nmf'] * 0.2,  # ÇOK DÜŞÜK AĞIRLIK
            self.matrices['numerical_robust'] * 0.3,  # DÜŞÜK AĞIRLIK
            self.matrices['kmeans_clusters'] * 0.1,  # MİNİMUM AĞIRLIK
            np.mean(self.matrices['cluster_distances'], axis=1).reshape(-1, 1) * 0.05  # MİNİMUM
        ]

        self.final_feature_matrix = np.hstack(feature_components)

        # Final benzerlik modeli
        self.similarity_models['final'] = NearestNeighbors(
            n_neighbors=min(250, len(self.books_df)),
            metric='cosine',
            algorithm='brute'
        )
        self.similarity_models['final'].fit(self.final_feature_matrix)

        logger.info(f"✅ Final yüksek güven matrisi hazır: {self.final_feature_matrix.shape}")
        logger.info("🎯 Ağırlık dağılımı: Yazar(3.0) > Kategori(2.8) > Açıklama(1.0) > Diğer(<0.5)")

    def get_high_confidence_recommendations(self, book_title, n_recommendations=10):
        """YÜKSEK GÜVENİLİRLİK öneriler - Yazar ve Kategori Odaklı"""
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

        logger.info(f"🎯 Hedef kitap: '{target_book['title']}' - {target_book['author']} - {target_book['category']}")

        # YAZAR ve KATEGORİ odaklı çoklu benzerlik hesaplama
        recommendations = self._calculate_author_category_similarity(target_idx, n_recommendations * 4)

        # GÜÇLENDIRILMIŞ güven skoru hesaplama
        recommendations = self._calculate_boosted_confidence_scores(recommendations, target_book)

        # Yazar-Kategori odaklı çeşitlilik optimizasyonu
        recommendations = self._optimize_author_category_diversity(recommendations, target_book)

        # Final filtreleme ve sıralama - GÜVENİLİRLİK ODAKLI
        final_recommendations = self._confidence_focused_final_ranking(
            recommendations, target_book, n_recommendations
        )

        return final_recommendations

    def _calculate_author_category_similarity(self, target_idx, n_candidates):
        """YAZAR ve KATEGORİ odaklı benzerlik hesaplama"""
        target_book = self.books_df.iloc[target_idx]

        # Farklı benzerlik türleri
        similarities = {}

        # 1. YAZAR benzerliği (EN ÖNEMLİ)
        target_author_features = self.matrices['author_enhanced_svd'][target_idx].reshape(1, -1)
        author_distances, author_indices = self.similarity_models['knn_author_enhanced'].kneighbors(
            target_author_features, n_neighbors=min(n_candidates + 100, len(self.books_df))
        )
        similarities['author'] = {
            idx: (1 - dist) * 1.2 for idx, dist in zip(author_indices[0][1:], author_distances[0][1:])  # Bonus
        }

        # 2. KATEGORİ benzerliği (YÜKSEK ÖNEMLİ)
        target_category_features = self.matrices['category_enhanced_svd'][target_idx].reshape(1, -1)
        category_distances, category_indices = self.similarity_models['knn_category_enhanced'].kneighbors(
            target_category_features, n_neighbors=min(n_candidates + 100, len(self.books_df))
        )
        similarities['category'] = {
            idx: (1 - dist) * 1.1 for idx, dist in zip(category_indices[0][1:], category_distances[0][1:])  # Bonus
        }

        # 3. AÇIKLAMA benzerliği (ORTA ÖNEMLİ)
        target_desc_features = self.matrices['description_svd'][target_idx].reshape(1, -1)
        desc_distances, desc_indices = self.similarity_models['knn_description'].kneighbors(
            target_desc_features, n_neighbors=min(n_candidates + 50, len(self.books_df))
        )
        similarities['description'] = {
            idx: 1 - dist for idx, dist in zip(desc_indices[0][1:], desc_distances[0][1:])
        }

        # 4. Sayısal özellik benzerliği
        target_numerical_features = self.matrices['numerical_robust'][target_idx].reshape(1, -1)
        num_distances, num_indices = self.similarity_models['knn_numerical'].kneighbors(
            target_numerical_features, n_neighbors=min(n_candidates + 50, len(self.books_df))
        )
        similarities['numerical'] = {
            idx: 1 - dist for idx, dist in zip(num_indices[0][1:], num_distances[0][1:])
        }

        # Tüm adayları birleştir
        all_candidates = set()
        for sim_dict in similarities.values():
            all_candidates.update(sim_dict.keys())

        recommendations = []
        for idx in all_candidates:
            if idx == target_idx:
                continue

            book = self.books_df.iloc[idx]
            same_author = book['author'] == target_book['author']
            same_category = book['category'] == target_book['category']

            # Benzerlik skorları
            author_sim = similarities['author'].get(idx, 0)
            category_sim = similarities['category'].get(idx, 0)
            description_sim = similarities['description'].get(idx, 0)
            numerical_sim = similarities['numerical'].get(idx, 0)

            # YAZAR ve KATEGORİ ÖNCELİKLİ dinamik ağırlık sistemi
            if same_author and same_category:
                # Aynı yazar ve kategori - maksimum güven
                weights = {
                    'author': 0.50,  # EN YÜKSEK
                    'category': 0.30,  # YÜKSEK
                    'description': 0.15,  # ORTA
                    'numerical': 0.05  # DÜŞÜK
                }
            elif same_author:
                # Aynı yazar - yüksek güven
                weights = {
                    'author': 0.55,  # EN YÜKSEK
                    'category': 0.20,  # ORTA
                    'description': 0.20,  # ORTA
                    'numerical': 0.05  # DÜŞÜK
                }
            elif same_category:
                # Aynı kategori - orta-yüksek güven
                weights = {
                    'author': 0.25,  # ORTA
                    'category': 0.45,  # EN YÜKSEK
                    'description': 0.25,  # ORTA
                    'numerical': 0.05  # DÜŞÜK
                }
            elif author_sim > 0.8 or category_sim > 0.8:
                # Yüksek yazar veya kategori benzerliği
                weights = {
                    'author': 0.40,  # YÜKSEK
                    'category': 0.35,  # YÜKSEK
                    'description': 0.20,  # ORTA
                    'numerical': 0.05  # DÜŞÜK
                }
            else:
                # Genel durumlar - YAZAR ve KATEGORİ hala öncelikli
                weights = {
                    'author': 0.35,  # YÜKSEK
                    'category': 0.30,  # YÜKSEK
                    'description': 0.25,  # ORTA
                    'numerical': 0.10  # DÜŞÜK
                }

            combined_similarity = (
                    author_sim * weights['author'] +
                    category_sim * weights['category'] +
                    description_sim * weights['description'] +
                    numerical_sim * weights['numerical']
            )

            # YÜKSEK GÜVENİLİRLİK bonusları
            same_author_bonus = 0.25 if same_author else 0  # ARTIRILD
            same_category_bonus = 0.20 if same_category else 0  # ARTIRILD

            # Yazar otoritesi bonusu
            author_authority_bonus = book['author_authority'] * 0.15  # YENİ BONUS

            # Kategori tutarlılık bonusu
            category_coherence_bonus = book['category_coherence'] * 0.10  # YENİ BONUS

            # Uzmanlık bonusu
            specialization_bonus = book['author_category_specialization'] * 0.10  # YENİ BONUS

            final_similarity = min(1.0, combined_similarity + same_author_bonus +
                                   same_category_bonus + author_authority_bonus +
                                   category_coherence_bonus + specialization_bonus)

            recommendations.append({
                'book_data': book,
                'similarity_score': final_similarity,
                'author_similarity': author_sim,
                'category_similarity': category_sim,
                'description_similarity': description_sim,
                'numerical_similarity': numerical_sim,
                'index': idx,
                'same_author': same_author,
                'same_category': same_category
            })

        # Benzerlik skoruna göre sırala
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations[:n_candidates]

    def _calculate_boosted_confidence_scores(self, recommendations, target_book):
        """GÜÇLENDIRILMIŞ güven skorları - Yazar ve Kategori Odaklı - GÜVENİLİRLİK ARTIRMA"""
        for rec in recommendations:
            book_data = rec['book_data']
            book_index = rec['index']

            # YAZAR ve KATEGORİ temelli güven faktörleri - EN YÜKSEK AĞIRLIK
            author_confidence = book_data['author_authority'] * 1.8  # BÜYÜK ARTIRILD
            category_confidence = book_data['category_coherence'] * 1.6  # BÜYÜK ARTIRILD
            specialization_confidence = book_data['author_category_specialization'] * 1.4  # ARTIRILD

            # Temel güven faktörleri - AĞIRLIK AZALTILDI
            rating_confidence = min(1.0, np.log1p(book_data['total_ratings']) / 8) * 0.5  # DAHA DA AZALTILD
            quality_confidence = book_data['publisher_quality'] * 0.4  # DAHA DA AZALTILD

            # NLP güven faktörleri - ÇOK DÜŞÜK AĞIRLIK
            semantic_confidence = book_data.get('semantic_density', 0) * 0.15
            topic_confidence = book_data.get('topic_coherence', 0) * 0.15
            text_quality_confidence = book_data.get('text_quality', 0) * 0.1

            # YAZAR ve KATEGORİ eşleşme bonusları - BÜYÜK BONUS
            same_author_confidence_bonus = 0.50 if rec['same_author'] else 0  # BÜYÜK ARTIRILD
            same_category_confidence_bonus = 0.40 if rec['same_category'] else 0  # BÜYÜK ARTIRILD

            # Yüksek benzerlik bonusları - YENİ
            high_author_similarity_bonus = 0.30 if rec['author_similarity'] > 0.8 else 0.15 if rec[
                                                                                                   'author_similarity'] > 0.6 else 0
            high_category_similarity_bonus = 0.25 if rec['category_similarity'] > 0.8 else 0.12 if rec[
                                                                                                       'category_similarity'] > 0.6 else 0

            # Kategori uyumluluğu güveni - ARTIRILD
            category_compatibility = self._get_enhanced_category_compatibility(
                target_book['category'], book_data['category']
            ) * 1.2  # ARTIRILD

            # Anomali skoru - AZALTILD
            hybrid_features = self.matrices['high_confidence_features'][book_index].reshape(1, -1)
            anomaly_score = self.confidence_models['isolation_forest'].decision_function(hybrid_features)[0]
            anomaly_confidence = max(0, (anomaly_score + 0.5) / 1.0) * 0.3  # AZALTILD

            # ML güven skoru - TUTARLI özellik vektörü kullan
            try:
                confidence_features = self._get_confidence_features(book_index).reshape(1, -1)

                # Özellik boyutu kontrolü
                if confidence_features.shape[1] != self.confidence_feature_dim:
                    if confidence_features.shape[1] < self.confidence_feature_dim:
                        padding = np.zeros((1, self.confidence_feature_dim - confidence_features.shape[1]))
                        confidence_features = np.hstack([confidence_features, padding])
                    else:
                        confidence_features = confidence_features[:, :self.confidence_feature_dim]

                ml_confidence = self.confidence_models['rf_confidence'].predict(confidence_features)[
                                    0] * 0.6  # AZALTILD
            except Exception as e:
                ml_confidence = 0.6  # Daha yüksek varsayılan değer

            # YÜKSEK GÜVENİLİRLİK kompozit skoru - YAZAR ve KATEGORİ ÖNCELİKLİ
            composite_confidence = (
                    author_confidence * 0.35 +  # EN YÜKSEK AĞIRLIK - ARTIRILD
                    category_confidence * 0.30 +  # ÇOK YÜKSEK AĞIRLIK - ARTIRILD
                    specialization_confidence * 0.20 +  # YÜKSEK AĞIRLIK - ARTIRILD
                    same_author_confidence_bonus * 0.08 +  # YÜKSEK BONUS
                    same_category_confidence_bonus * 0.05 +  # ORTA BONUS
                    high_author_similarity_bonus * 0.015 +  # YENİ BONUS
                    high_category_similarity_bonus * 0.01 +  # YENİ BONUS
                    category_compatibility * 0.005  # DÜŞÜK AĞIRLIK
            )

            # Minimum güven skoru garantisi - YENİ
            base_confidence = 0.3  # Temel güven seviyesi
            if rec['same_author']:
                base_confidence = 0.6  # Aynı yazar için daha yüksek
            elif rec['same_category']:
                base_confidence = 0.5  # Aynı kategori için yüksek
            elif rec['author_similarity'] > 0.7 or rec['category_similarity'] > 0.7:
                base_confidence = 0.45  # Yüksek benzerlik için

            # Güven skoru normalizasyonu ve büyük artırma
            final_confidence = min(1.0, max(base_confidence, composite_confidence * 1.8))  # %80 artış + minimum garanti

            # Ekstra bonus sistemi - YENİ
            extra_bonus = 0
            if rec['same_author'] and rec['same_category']:
                extra_bonus = 0.15  # Süper bonus
            elif rec['same_author'] and rec['category_similarity'] > 0.7:
                extra_bonus = 0.10  # Yüksek bonus
            elif rec['same_category'] and rec['author_similarity'] > 0.7:
                extra_bonus = 0.08  # Yüksek bonus

            final_confidence = min(1.0, final_confidence + extra_bonus)

            rec['confidence_score'] = final_confidence
            rec['confidence_details'] = {
                'author_confidence': author_confidence,
                'category_confidence': category_confidence,
                'specialization_confidence': specialization_confidence,
                'same_author_bonus': same_author_confidence_bonus,
                'same_category_bonus': same_category_confidence_bonus,
                'high_author_similarity_bonus': high_author_similarity_bonus,
                'high_category_similarity_bonus': high_category_similarity_bonus,
                'category_compatibility': category_compatibility,
                'base_confidence': base_confidence,
                'extra_bonus': extra_bonus,
                'rating_confidence': rating_confidence,
                'quality_confidence': quality_confidence,
                'semantic_confidence': semantic_confidence,
                'topic_confidence': topic_confidence,
                'anomaly_confidence': anomaly_confidence,
                'ml_confidence': ml_confidence
            }

        return recommendations

    def _optimize_author_category_diversity(self, recommendations, target_book):
        """Yazar-Kategori odaklı çeşitlilik optimizasyonu - GÜVENİLİRLİK ARTIRMA"""
        # DAHA DÜŞÜK güven eşiği - daha fazla önerinin geçmesi için
        min_confidence = max(0.60, self.hyperparameters['confidence_threshold'] - 0.20)  # DAHA DÜŞÜK

        # Aynı yazar veya kategori için çok daha esnek güven eşiği
        high_confidence_recs = []
        for rec in recommendations:
            if (rec['confidence_score'] >= min_confidence or
                    rec['same_author'] or rec['same_category'] or
                    rec['similarity_score'] > 0.7 or  # DÜŞÜRÜLDÜ
                    rec['author_similarity'] > 0.6 or  # YENİ KRITER
                    rec['category_similarity'] > 0.6):  # YENİ KRITER
                high_confidence_recs.append(rec)

        if len(high_confidence_recs) < 10:  # ARTIRILD
            # En iyi skorluları al - YAZAR ve KATEGORİ çok daha öncelikli
            high_confidence_recs = sorted(recommendations,
                                          key=lambda x: (
                                                  x['author_similarity'] * 0.50 +  # BÜYÜK ARTIRILD
                                                  x['category_similarity'] * 0.45 +  # BÜYÜK ARTIRILD
                                                  x['confidence_score'] * 0.05  # BÜYÜK AZALTILD
                                          ),
                                          reverse=True)[:15]  # ARTIRILD

        # Çeşitlilik kontrolü - YAZAR ve KATEGORİ ÇOK DAHA ODAKLI
        diverse_recommendations = []
        used_authors = set()
        used_categories = set()
        same_author_count = 0
        same_category_count = 0

        # Sıralama: YAZAR ve KATEGORİ maksimum öncelik
        high_confidence_recs.sort(
            key=lambda x: (
                    x['author_similarity'] * 0.45 +  # ARTIRILD
                    x['category_similarity'] * 0.40 +  # ARTIRILD
                    x['confidence_score'] * 0.10 +  # AZALTILD
                    x['similarity_score'] * 0.05  # AZALTILD
            ),
            reverse=True
        )

        for rec in high_confidence_recs:
            book_data = rec['book_data']

            same_author = rec['same_author']
            same_category = rec['same_category']

            # YAZAR ve KATEGORİ için çok daha esnek çeşitlilik kuralları
            if same_author and same_author_count < 8:  # ARTIRILD
                same_author_count += 1
                author_diversity_bonus = 0.25  # ARTIRILD
            elif book_data['author'] not in used_authors:
                author_diversity_bonus = 0.15  # ARTIRILD
            else:
                author_diversity_bonus = 0  # CEZA KALDIRILDI

            if same_category and same_category_count < 10:  # ARTIRILD
                same_category_count += 1
                category_diversity_bonus = 0.20  # ARTIRILD
            elif book_data['category'] not in used_categories:
                category_diversity_bonus = 0.12  # ARTIRILD
            else:
                category_diversity_bonus = 0  # CEZA KALDIRILDI

            # Çeşitlilik skoru - daha cömert
            diversity_score = 1.0 + author_diversity_bonus + category_diversity_bonus

            # YAZAR ve KATEGORİ ÖNCELİKLİ final skor - daha yüksek ağırlıklar
            rec['diversity_score'] = diversity_score
            rec['final_score'] = (
                    rec['author_similarity'] * 0.40 +  # EN YÜKSEK AĞIRLIK - ARTIRILD
                    rec['category_similarity'] * 0.35 +  # ÇOK YÜKSEK AĞIRLIK - ARTIRILD
                    rec['confidence_score'] * 0.15 +  # ORTA AĞIRLIK - AZALTILD
                    rec['similarity_score'] * 0.08 +  # DÜŞÜK AĞIRLIK - AZALTILD
                    rec['diversity_score'] * 0.02  # ÇOK DÜŞÜK AĞIRLIK - AZALTILD
            )

            diverse_recommendations.append(rec)

            # Kullanılan değerleri güncelle
            used_authors.add(book_data['author'])
            used_categories.add(book_data['category'])

        return diverse_recommendations

    def _get_enhanced_category_compatibility(self, target_category, candidate_category):
        """Gelişmiş kategori uyumluluğu - GÜVENİLİRLİK ARTIRMA"""
        if target_category == candidate_category:
            return 1.0  # Tam eşleşme

        if target_category in self.genre_compatibility:
            compatibility = self.genre_compatibility[target_category].get(candidate_category, 0.05)
            return min(0.95, compatibility * 5)  # BÜYÜK ARTIRILD

        # Benzer kelimeleri kontrol et - YENİ
        target_words = set(target_category.lower().split())
        candidate_words = set(candidate_category.lower().split())

        if target_words & candidate_words:  # Ortak kelime varsa
            return 0.6  # Yüksek uyumluluk

        # Basit kategori grupları - YENİ
        fiction_categories = {'roman', 'hikaye', 'edebiyat', 'kurgu', 'fiction'}
        non_fiction_categories = {'tarih', 'biyografi', 'bilim', 'felsefe', 'history', 'biography'}
        fantasy_categories = {'fantastik', 'fantasy', 'bilimkurgu', 'scifi', 'büyü'}

        target_lower = target_category.lower()
        candidate_lower = candidate_category.lower()

        # Aynı grup içindeyse
        for category_group in [fiction_categories, non_fiction_categories, fantasy_categories]:
            if (any(word in target_lower for word in category_group) and
                    any(word in candidate_lower for word in category_group)):
                return 0.4  # Orta-yüksek uyumluluk

        return 0.1  # Minimum uyumluluk - ARTIRILD

    def _confidence_focused_final_ranking(self, recommendations, target_book, n_recommendations):
        """GÜVENİLİRLİK odaklı final sıralama - Yazar ve Kategori Öncelikli"""
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

                # Skorlar - YAZAR ve KATEGORİ ÖNCELİKLİ
                'similarity_score': round(rec['similarity_score'], 4),
                'confidence_score': round(rec['confidence_score'], 4),
                'diversity_score': round(rec['diversity_score'], 4),
                'final_score': round(rec['final_score'], 4),

                # Alt benzerlik skorları - YAZAR ve KATEGORİ ÖNCELİKLİ
                'author_similarity': round(rec['author_similarity'], 4),
                'category_similarity': round(rec['category_similarity'], 4),
                'description_similarity': round(rec['description_similarity'], 4),
                'numerical_similarity': round(rec['numerical_similarity'], 4),

                # Güven detayları
                'confidence_details': {
                    k: round(v, 4) for k, v in rec['confidence_details'].items()
                },

                # Eşleşme bilgileri
                'same_author': rec['same_author'],
                'same_category': rec['same_category'],

                # Ek bilgiler
                'recommendation_reasons': self._generate_confidence_focused_reasons(rec, target_book)
            }

            final_recommendations.append(final_rec)

        return final_recommendations

    def _generate_confidence_focused_reasons(self, rec, target_book):
        """GÜVENİLİRLİK odaklı öneri nedenlerini oluştur - Yazar ve Kategori Öncelikli"""
        reasons = []
        book_data = rec['book_data']

        # YAZAR ve KATEGORİ eşleşmeleri - EN ÖNCELİKLİ
        if rec['same_author']:
            reasons.append(f"Aynı yazar: {book_data['author']}")
        elif rec.get('author_similarity', 0) > 0.8:
            reasons.append("Çok benzer yazım tarzı")
        elif rec.get('author_similarity', 0) > 0.6:
            reasons.append("Benzer yazar profili")

        if rec['same_category']:
            reasons.append(f"Aynı kategori: {book_data['category']}")
        elif rec.get('category_similarity', 0) > 0.8:
            reasons.append("Çok benzer tür")
        elif rec.get('category_similarity', 0) > 0.6:
            reasons.append("Benzer kategori")

        # Otorite ve kalite göstergeleri - YÜKSEK ÖNCELİK
        if book_data['author_authority'] > 0.8:
            reasons.append("Çok güvenilir yazar")
        elif book_data['author_authority'] > 0.6:
            reasons.append("Güvenilir yazar")

        if book_data['author_category_specialization'] > 0.7:
            reasons.append("Bu kategoride uzman yazar")

        # Rating ve popülerlik - ORTA ÖNCELİK
        if book_data['average_rating'] > 4.2 and book_data['total_ratings'] > 500:
            reasons.append("Çok yüksek kullanıcı puanı")
        elif book_data['average_rating'] > 3.8 and book_data['total_ratings'] > 100:
            reasons.append("Yüksek kullanıcı puanı")

        # İçerik benzerliği - DÜŞÜK ÖNCELİK
        if rec.get('description_similarity', 0) > 0.8:
            reasons.append("Çok benzer içerik")
        elif rec.get('description_similarity', 0) > 0.6:
            reasons.append("Benzer konu ve tema")

        # NLP özellikleri - EN DÜŞÜK ÖNCELİK
        if book_data.get('semantic_density', 0) > 0.7:
            reasons.append("Zengin içerik")

        return reasons[:4]  # En fazla 4 neden

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

    def get_model_statistics(self):
        """Model istatistiklerini göster"""
        if not self.is_trained:
            logger.error("❌ Model henüz eğitilmedi!")
            return None

        stats = {
            'total_books': len(self.books_df),
            'unique_authors': self.books_df['author'].nunique(),
            'unique_categories': self.books_df['category'].nunique(),
            'unique_publishers': self.books_df['publisher'].nunique(),
            'feature_matrix_shape': self.final_feature_matrix.shape,
            'confidence_threshold': self.hyperparameters['confidence_threshold'],
            'author_category_focused': True,
            'high_confidence_model': True,
            'model_components': {
                'vectorizers': len(self.vectorizers),
                'dimension_reducers': len(self.dimension_reducers),
                'similarity_models': len(self.similarity_models),
                'confidence_models': len(self.confidence_models)
            },
            'feature_weights': {
                'author_weight': 3.0,
                'category_weight': 2.8,
                'description_weight': 1.0,
                'combined_weight': 0.4,
                'numerical_weight': 0.3
            }
        }

        logger.info("📊 Yüksek Güvenilirlik Model İstatistikleri:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")

        return stats

    def test_recommendation_quality(self, test_books, n_recommendations=5):
        """Öneri kalitesini test et - GÜVENİLİRLİK ODAKLI"""
        if not self.is_trained:
            logger.error("❌ Model henüz eğitilmedi!")
            return

        logger.info("🧪 YÜKSEK GÜVENİLİRLİK Öneri kalitesi testi başlıyor...")

        total_confidence = 0
        total_similarity = 0
        total_author_similarity = 0
        total_category_similarity = 0
        successful_recommendations = 0

        for book_title in test_books:
            logger.info(f"\n🔍 Test: '{book_title}'")
            recommendations = self.get_high_confidence_recommendations(
                book_title, n_recommendations
            )

            if recommendations:
                successful_recommendations += 1
                avg_confidence = np.mean([rec['confidence_score'] for rec in recommendations])
                avg_similarity = np.mean([rec['similarity_score'] for rec in recommendations])
                avg_author_sim = np.mean([rec['author_similarity'] for rec in recommendations])
                avg_category_sim = np.mean([rec['category_similarity'] for rec in recommendations])

                total_confidence += avg_confidence
                total_similarity += avg_similarity
                total_author_similarity += avg_author_sim
                total_category_similarity += avg_category_sim

                logger.info(
                    f"   ✅ {len(recommendations)} öneri - Güven: {avg_confidence:.3f}, "
                    f"Benzerlik: {avg_similarity:.3f}, Yazar: {avg_author_sim:.3f}, "
                    f"Kategori: {avg_category_sim:.3f}")

                # En iyi 3 öneriyi göster
                for i, rec in enumerate(recommendations[:3]):
                    logger.info(f"   {i + 1}. {rec['title']} - {rec['author']} "
                                f"(Güven: {rec['confidence_score']:.3f}, "
                                f"Yazar: {rec['author_similarity']:.3f}, "
                                f"Kategori: {rec['category_similarity']:.3f})")
            else:
                logger.warning(f"   ❌ '{book_title}' için öneri bulunamadı!")

        if successful_recommendations > 0:
            avg_total_confidence = total_confidence / successful_recommendations
            avg_total_similarity = total_similarity / successful_recommendations
            avg_total_author_sim = total_author_similarity / successful_recommendations
            avg_total_category_sim = total_category_similarity / successful_recommendations
            success_rate = successful_recommendations / len(test_books)

            logger.info(f"\n📈 YÜKSEK GÜVENİLİRLİK Test Sonuçları:")
            logger.info(f"   Başarı Oranı: {success_rate:.2%}")
            logger.info(f"   Ortalama Güven Skoru: {avg_total_confidence:.3f}")
            logger.info(f"   Ortalama Genel Benzerlik: {avg_total_similarity:.3f}")
            logger.info(f"   Ortalama Yazar Benzerlik: {avg_total_author_sim:.3f}")
            logger.info(f"   Ortalama Kategori Benzerlik: {avg_total_category_sim:.3f}")

            return {
                'success_rate': success_rate,
                'avg_confidence': avg_total_confidence,
                'avg_similarity': avg_total_similarity,
                'avg_author_similarity': avg_total_author_sim,
                'avg_category_similarity': avg_total_category_sim,
                'successful_tests': successful_recommendations,
                'total_tests': len(test_books)
            }

        return None


def main():
    """Ana fonksiyon - YÜKSEK GÜVENİLİRLİK Model"""
    # Veritabanı bağlantısı
    db_manager = DatabaseManager()

    try:
        # Veri çekme
        books_df = db_manager.get_books_dataframe()
        if books_df.empty:
            logger.error("❌ Veri çekilemedi!")
            return

        logger.info(f"📚 Toplam {len(books_df)} kitap verisi çekildi")

        # YÜKSEK GÜVENİLİRLİK model oluşturma ve eğitme
        model = HighConfidenceBookRecommendationModel()
        model.train_high_confidence_model(books_df)

        # Model istatistiklerini göster
        model.get_model_statistics()

        # Test kitapları - YAZAR ve KATEGORİ odaklı testler
        test_books = [
            'yüzüklerin efendisi',  # Fantastik - ünlü yazar
            'harry potter',  # Fantastik - çok popüler yazar
            '1984',  # Distopya - klasik yazar
            'görmek',  # Türk edebiyatı
            'uçurtma avcısı',  # Drama - tanınmış yazar
            'hayvan çiftliği',  # Alegorik - ünlü yazar
            'simyacı',  # Felsefe/Roman
            'suç ve ceza'  # Klasik edebiyat
        ]

        # Kalite testi - GÜVENİLİRLİK ODAKLI
        quality_results = model.test_recommendation_quality(test_books, n_recommendations=6)

        # Detaylı öneriler - YAZAR ve KATEGORİ ANALİZİ
        for book_title in test_books[:4]:  # İlk 4 kitap için detaylı
            print(f"\n{'=' * 120}")
            print(f"🔍 '{book_title}' için YÜKSEK GÜVENİLİRLİK önerileri:")
            print('=' * 120)

            recommendations = model.get_high_confidence_recommendations(
                book_title, n_recommendations=8
            )

            if recommendations:
                for rec in recommendations:
                    print(f"\n{rec['rank']}. 📖 {rec['title']}")
                    print(f"   👤 Yazar: {rec['author']}")
                    print(f"   📂 Kategori: {rec['category']}")
                    print(f"   📊 Puan: {rec['average_rating']:.1f} ({rec['total_ratings']} değerlendirme)")
                    print(f"   🎯 Final Skor: {rec['final_score']:.4f}")
                    print(f"   🔒 Güven Skoru: {rec['confidence_score']:.4f}")

                    # YAZAR ve KATEGORİ ANALİZİ - ÖNEMLİ
                    print(f"   📊 Benzerlik Analizi:")
                    print(f"      👨‍✍️ Yazar Benzerlik: {rec['author_similarity']:.4f}")
                    print(f"      🏷️ Kategori Benzerlik: {rec['category_similarity']:.4f}")
                    print(f"      📝 İçerik Benzerlik: {rec['description_similarity']:.4f}")
                    print(f"      🔢 Sayısal Benzerlik: {rec['numerical_similarity']:.4f}")

                    # Eşleşme durumu
                    matches = []
                    if rec['same_author']:
                        matches.append("✅ AYNI YAZAR")
                    if rec['same_category']:
                        matches.append("✅ AYNI KATEGORİ")
                    if matches:
                        print(f"   🎯 Eşleşmeler: {' | '.join(matches)}")

                    print(f"   💡 Öneri Nedenleri: {', '.join(rec['recommendation_reasons'])}")

                    # Güven detayları (sadece ilk 3 için)
                    if rec['rank'] <= 3:
                        print(f"   🔒 Güven Analizi:")
                        important_details = [
                            'author_confidence', 'category_confidence', 'specialization_confidence',
                            'same_author_bonus', 'same_category_bonus'
                        ]
                        for detail_key in important_details:
                            if detail_key in rec['confidence_details']:
                                detail_value = rec['confidence_details'][detail_key]
                                print(f"      - {detail_key}: {detail_value:.3f}")

                    # Açıklama önizlemesi (sadece ilk 2 için)
                    if rec['rank'] <= 2:
                        print(f"   📜 Açıklama: {rec['description'][:120]}...")
            else:
                print("❌ Öneri bulunamadı!")

        # Model kaydetme
        model_filename = 'high_confidence_author_category_book_model_v4.pkl'
        model.save_model(model_filename)

        # YÜKSEK GÜVENİLİRLİK Özet rapor
        if quality_results:
            print(f"\n{'=' * 120}")
            print("📈 YÜKSEK GÜVENİLİRLİK MODEL PERFORMANS RAPORU")
            print('=' * 120)
            print(f"✅ Başarı Oranı: {quality_results['success_rate']:.1%}")
            print(f"🔒 Ortalama Güven Skoru: {quality_results['avg_confidence']:.3f}")
            print(f"🔗 Ortalama Genel Benzerlik: {quality_results['avg_similarity']:.3f}")
            print(f"👨‍✍️ Ortalama Yazar Benzerlik: {quality_results['avg_author_similarity']:.3f}")
            print(f"🏷️ Ortalama Kategori Benzerlik: {quality_results['avg_category_similarity']:.3f}")
            print(f"📊 Başarılı Test: {quality_results['successful_tests']}/{quality_results['total_tests']}")
            print(f"💾 Model kaydedildi: {model_filename}")

            # GÜVENİLİRLİK değerlendirmesi
            if quality_results['avg_confidence'] > 0.80:
                print(f"🎉 MÜKEMMEL: Güven skoru hedefin üzerinde!")
            elif quality_results['avg_confidence'] > 0.75:
                print(f"✅ İYİ: Güven skoru hedef aralıkta!")
            elif quality_results['avg_confidence'] > 0.65:
                print(f"⚠️ KABUL EDİLEBİLİR: Güven skoru kabul edilebilir seviyede!")
            else:
                print(f"❌ DİKKAT: Güven skoru hedefin altında!")

        # Özel güvenilirlik test senaryoları
        print(f"\n{'=' * 120}")
        print("🧪 ÖZEL GÜVENİLİRLİK TEST SENARYOLARI")
        print('=' * 120)

        special_tests = [
            ('yüzüklerin efendisi', 'hobbit'),  # Aynı yazar test
            ('harry potter', 'chronicles'),  # Aynı tür test
            ('1984', 'cesur yeni dünya'),  # Distopya türü test
        ]

        for source_book, expected_similar in special_tests:
            print(f"\n🔬 Test: '{source_book}' -> '{expected_similar}' beklentisi")
            recommendations = model.get_high_confidence_recommendations(source_book, 10)

            if recommendations:
                found_expected = False
                for i, rec in enumerate(recommendations):
                    if expected_similar.lower() in rec['title'].lower():
                        print(f"   ✅ '{expected_similar}' {i + 1}. sırada bulundu!")
                        print(f"      Güven Skoru: {rec['confidence_score']:.4f}")
                        print(f"      Yazar Benzerlik: {rec['author_similarity']:.4f}")
                        print(f"      Kategori Benzerlik: {rec['category_similarity']:.4f}")
                        found_expected = True
                        break

                if not found_expected:
                    print(f"   ❌ '{expected_similar}' ilk 10'da bulunamadı!")
                    print("   💡 En yüksek güven skorlu öneriler:")
                    for i, rec in enumerate(recommendations[:3]):
                        print(f"      {i + 1}. {rec['title']} (Güven: {rec['confidence_score']:.3f}, "
                              f"Yazar: {rec['author_similarity']:.3f}, "
                              f"Kategori: {rec['category_similarity']:.3f})")

    except Exception as e:
        logger.error(f"❌ Ana fonksiyon hatası: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db_manager.close_connection()


if __name__ == "__main__":
    main()