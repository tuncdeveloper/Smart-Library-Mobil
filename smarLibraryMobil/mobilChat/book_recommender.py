from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pickle
import logging
import os
from contextlib import asynccontextmanager

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model değişkeni
model = None


# Pydantic modelleri
class BookRecommendation(BaseModel):
    rank: int
    title: str
    author: str
    category: str
    publisher: str
    publication_year: int
    average_rating: float
    total_ratings: int
    page_count: Optional[int]
    description: str
    similarity_score: float
    confidence_score: float
    diversity_score: float
    final_score: float
    confidence_details: Dict[str, float]
    recommendation_reasons: List[str]


class RecommendationRequest(BaseModel):
    book_title: str
    n_recommendations: Optional[int] = 5


class RecommendationResponse(BaseModel):
    success: bool
    message: str
    target_book: Optional[Dict[str, Any]] = None
    recommendations: List[BookRecommendation]
    total_recommendations: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str


# Model yükleme fonksiyonu
def load_recommendation_model(model_path: str):
    """Öneri modelini yükle"""
    global model
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        logger.info(f"📚 Model yükleniyor: {model_path}")

        # Model dosyasını import et
        from BookModelsLearning import UltraHighConfidenceBookRecommendationModel

        model = UltraHighConfidenceBookRecommendationModel()
        model.load_model(model_path)

        logger.info("✅ Model başarıyla yüklendi!")
        return True

    except Exception as e:
        logger.error(f"❌ Model yükleme hatası: {e}")
        return False


# Startup ve shutdown event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 FastAPI başlıyor...")
    model_path = "ultra_high_confidence_book_model_v4.pkl"

    success = load_recommendation_model(model_path)
    if not success:
        logger.error("❌ Model yüklenemedi! Uygulama çalışmayabilir.")

    yield

    # Shutdown
    logger.info("🛑 FastAPI kapatılıyor...")


# FastAPI uygulaması
app = FastAPI(
    title="Ultra High Confidence Book Recommendation API",
    description="José Saramago tarzı kitaplar için gelişmiş yapay zeka tabanlı kitap öneri sistemi",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Üretimde belirli domainleri belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Sağlık durumu kontrolü"""
    return HealthResponse(
        status="healthy" if model and model.is_trained else "unhealthy",
        model_loaded=model is not None and model.is_trained,
        message="API çalışıyor" if model and model.is_trained else "Model yüklenmedi"
    )


# Ana öneri endpoint'i
@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Kitap önerileri al

    - **book_title**: Öneri alınacak kitap başlığı
    - **n_recommendations**: Öneri sayısı (varsayılan: 5, maksimum: 20)
    """
    global model

    # Model kontrolü
    if not model or not model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi veya eğitilmedi"
        )

    # Parametre validasyonu
    book_title = request.book_title.strip()
    if not book_title:
        raise HTTPException(
            status_code=400,
            detail="Kitap başlığı boş olamaz"
        )

    n_recommendations = min(max(request.n_recommendations, 1), 20)

    try:
        logger.info(f"🔍 '{book_title}' için {n_recommendations} öneri isteniyor...")

        # Önerileri al
        recommendations = model.get_ultra_high_confidence_recommendations(
            book_title=book_title,
            n_recommendations=n_recommendations
        )

        if not recommendations:
            return RecommendationResponse(
                success=False,
                message=f"'{book_title}' kitabı bulunamadı veya öneri üretilemedi",
                recommendations=[],
                total_recommendations=0
            )

        # Hedef kitap bilgisini al
        target_book_data = None
        book_matches = model.books_df[
            model.books_df['title'].str.contains(book_title, case=False, na=False)
        ]

        if not book_matches.empty:
            target_book = book_matches.iloc[0]
            target_book_data = {
                "title": target_book['title'],
                "author": target_book['author'],
                "category": target_book['category'],
                "publisher": target_book['publisher'],
                "publication_year": int(target_book['publication_year']),
                "average_rating": float(target_book['average_rating']),
                "total_ratings": int(target_book['total_ratings']),
                "description": target_book['description'][:200] + "..." if len(
                    str(target_book['description'])) > 200 else target_book['description']
            }

        # Pydantic modellerine dönüştür
        recommendation_models = []
        for rec in recommendations:
            recommendation_models.append(BookRecommendation(**rec))

        logger.info(f"✅ {len(recommendations)} öneri başarıyla döndürüldü")

        return RecommendationResponse(
            success=True,
            message=f"'{book_title}' için {len(recommendations)} öneri bulundu",
            target_book=target_book_data,
            recommendations=recommendation_models,
            total_recommendations=len(recommendations)
        )

    except Exception as e:
        logger.error(f"❌ Öneri alma hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Öneri sistemi hatası: {str(e)}"
        )


# GET endpoint'i de ekleyelim
@app.get("/recommendations/{book_title}", response_model=RecommendationResponse)
async def get_recommendations_get(
        book_title: str,
        n_recommendations: int = Query(default=5, ge=1, le=20, description="Öneri sayısı")
):
    """
    GET methodu ile kitap önerileri al

    - **book_title**: Öneri alınacak kitap başlığı (URL path parametresi)
    - **n_recommendations**: Öneri sayısı (query parametresi)
    """
    request = RecommendationRequest(
        book_title=book_title,
        n_recommendations=n_recommendations
    )
    return await get_recommendations(request)


# Model istatistikleri endpoint'i
@app.get("/model/stats")
async def get_model_stats():
    """Model istatistiklerini döndür"""
    global model

    if not model or not model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenmedi"
        )

    try:
        total_books = len(model.books_df) if model.books_df is not None else 0
        unique_authors = model.books_df['author'].nunique() if model.books_df is not None else 0
        unique_categories = model.books_df['category'].nunique() if model.books_df is not None else 0

        return {
            "total_books": total_books,
            "unique_authors": unique_authors,
            "unique_categories": unique_categories,
            "model_features": model.final_feature_matrix.shape[1] if hasattr(model, 'final_feature_matrix') else 0,
            "hyperparameters": model.hyperparameters if hasattr(model, 'hyperparameters') else {},
            "confidence_threshold": model.hyperparameters.get('confidence_threshold', 0.8) if hasattr(model,
                                                                                                      'hyperparameters') else 0.8
        }

    except Exception as e:
        logger.error(f"❌ İstatistik alma hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"İstatistik hatası: {str(e)}"
        )


# Kategori listesi endpoint'i
@app.get("/categories")
async def get_categories():
    """Mevcut kategorilerin listesini döndür"""
    global model

    if not model or not model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenmedi"
        )

    try:
        categories = sorted(model.books_df['category'].unique().tolist())
        return {
            "categories": categories,
            "total_categories": len(categories)
        }

    except Exception as e:
        logger.error(f"❌ Kategori alma hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Kategori hatası: {str(e)}"
        )


# Yazar listesi endpoint'i
@app.get("/authors")
async def get_authors():
    """Popüler yazarların listesini döndür"""
    global model

    if not model or not model.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenmedi"
        )

    try:
        # En popüler 50 yazarı döndür
        author_stats = model.books_df.groupby('author').agg({
            'id': 'count',
            'average_rating': 'mean',
            'total_ratings': 'sum'
        }).reset_index()

        author_stats.columns = ['author', 'book_count', 'avg_rating', 'total_ratings']
        author_stats = author_stats.sort_values('total_ratings', ascending=False).head(50)

        authors = []
        for _, row in author_stats.iterrows():
            authors.append({
                "name": row['author'],
                "book_count": int(row['book_count']),
                "average_rating": round(float(row['avg_rating']), 2),
                "total_ratings": int(row['total_ratings'])
            })

        return {
            "authors": authors,
            "total_authors": len(authors)
        }

    except Exception as e:
        logger.error(f"❌ Yazar alma hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Yazar hatası: {str(e)}"
        )


# Ana sayfa
@app.get("/")
async def root():
    """Ana sayfa"""
    return {
        "message": "Ultra High Confidence Book Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommendations_post": "/recommendations",
            "recommendations_get": "/recommendations/{book_title}",
            "model_stats": "/model/stats",
            "categories": "/categories",
            "authors": "/authors",
            "docs": "/docs"
        }
    }


# Uygulama çalıştırma
if __name__ == "__main__":
    import uvicorn

    # Geliştirme ortamı için
    uvicorn.run(
        "book_recommender:app",  # ← BURAYI DÜZELT
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        log_level="info"
    )