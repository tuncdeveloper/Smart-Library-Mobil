package com.smart.smartLibraryWeb.service;


import com.smart.smartLibraryWeb.config.BookRecommendationClient;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Slf4j
@Service
@RequiredArgsConstructor
public class BookRecommendationTestService {

    private final BookRecommendationClient client;

    /**
     * Senkron health check
     */
    public BookRecommendationClient.HealthResponse checkHealthSync() {
        try {
            return client.checkHealth().block();
        } catch (Exception e) {
            log.error("❌ Senkron health check hatası: {}", e.getMessage());
            return new BookRecommendationClient.HealthResponse("error", false, "Health check başarısız: " + e.getMessage());
        }
    }

    /**
     * Asenkron health check
     */
    public Mono<BookRecommendationClient.HealthResponse> checkHealthAsync() {
        return client.checkHealth();
    }

    /**
     * Kitap önerisi al (senkron)
     */
    public BookRecommendationClient.RecommendationResponse getRecommendationsSync(String bookTitle, Integer nRecommendations) {
        try {
            log.info("🔍 Senkron öneri isteniyor: '{}' için {} öneri", bookTitle, nRecommendations);
            return client.getRecommendationsGet(bookTitle, nRecommendations).block();
        } catch (Exception e) {
            log.error("❌ Senkron öneri hatası: {}", e.getMessage());
            return new BookRecommendationClient.RecommendationResponse(false, "Öneri alma başarısız: " + e.getMessage(),
                    null, null, 0);
        }
    }

    /**
     * Kitap önerisi al (asenkron)
     */
    public Mono<BookRecommendationClient.RecommendationResponse> getRecommendationsAsync(String bookTitle, Integer nRecommendations) {
        log.info("🔍 Asenkron öneri isteniyor: '{}' için {} öneri", bookTitle, nRecommendations);
        return client.getRecommendationsGet(bookTitle, nRecommendations);
    }

    /**
     * Model istatistikleri al (senkron)
     */
    public BookRecommendationClient.ModelStats getModelStatsSync() {
        try {
            return client.getModelStats().block();
        } catch (Exception e) {
            log.error("❌ Model istatistik hatası: {}", e.getMessage());
            return new BookRecommendationClient.ModelStats(0, 0, 0, 0, null, 0.0);
        }
    }

    /**
     * Tüm testleri çalıştır
     */
    public CompletableFuture<Object> runAllTestsAsync() {
        return client.generateTestReport()
                .toFuture()
                .handle((result, throwable) -> {
                    if (throwable != null) {
                        log.error("❌ Test raporu hatası: {}", throwable.getMessage());
                        return "Test raporu oluşturulamadı: " + throwable.getMessage();
                    }
                    return result;
                });
    }

    /**
     * Belirli kitaplar için toplu test
     */
    public CompletableFuture<Map<String, Object>> bulkTestBooks(String[] bookTitles) {
        log.info("📚 {} kitap için toplu test başlıyor...", bookTitles.length);

        return CompletableFuture.supplyAsync(() -> {
            Map<String, Object> results = new java.util.HashMap<>();

            for (String bookTitle : bookTitles) {
                try {
                    BookRecommendationClient.RecommendationResponse response = getRecommendationsSync(bookTitle, 3);
                    results.put(bookTitle, Map.of(
                            "success", response.getSuccess(),
                            "message", response.getMessage(),
                            "recommendation_count", response.getTotalRecommendations(),
                            "target_book", response.getTargetBook() != null ?
                                    response.getTargetBook().getTitle() : "Bulunamadı"
                    ));
                } catch (Exception e) {
                    results.put(bookTitle, Map.of(
                            "success", false,
                            "error", e.getMessage()
                    ));
                }
            }

            log.info("✅ Toplu test tamamlandı: {} kitap test edildi", bookTitles.length);
            return results;
        });
    }
}
