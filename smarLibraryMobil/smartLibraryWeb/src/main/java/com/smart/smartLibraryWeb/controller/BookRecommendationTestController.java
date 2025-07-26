package com.smart.smartLibraryWeb.controller;


import com.smart.smartLibraryWeb.config.BookRecommendationClient;

import com.smart.smartLibraryWeb.service.BookRecommendationTestService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Slf4j
@RestController
@RequestMapping("/api/test")
@RequiredArgsConstructor
public class BookRecommendationTestController {

    private final BookRecommendationTestService bookRecommendationTestService;

    /**
     * FastAPI health check
     */
    @GetMapping("/health")
    public ResponseEntity<BookRecommendationClient.HealthResponse> checkHealth() {
        log.info("🏥 Health check endpoint çağrıldı");
        BookRecommendationClient.HealthResponse health = bookRecommendationTestService.checkHealthSync();
        return ResponseEntity.ok(health);
    }

    /**
     * Asenkron health check
     */
    @GetMapping("/health/async")
    public Mono<ResponseEntity<BookRecommendationClient.HealthResponse>> checkHealthAsync() {
        log.info("🏥 Asenkron health check endpoint çağrıldı");
        return bookRecommendationTestService.checkHealthAsync()
                .map(ResponseEntity::ok);
    }

    /**
     * Kitap önerisi al
     */
    //@RequestMapping("/api/test")
    @GetMapping("/recommendations")
    public ResponseEntity<BookRecommendationClient.RecommendationResponse> getRecommendations(
            @RequestParam String bookTitle,
            @RequestParam(defaultValue = "5") Integer nRecommendations) {

        log.info("📚 Öneri endpoint çağrıldı: '{}' için {} öneri", bookTitle, nRecommendations);

        BookRecommendationClient.RecommendationResponse response = bookRecommendationTestService.getRecommendationsSync(bookTitle, nRecommendations);
        return ResponseEntity.ok(response);
    }

    /**
     * Asenkron kitap önerisi
     */

    @GetMapping("/recommendations/async")
    public Mono<ResponseEntity<BookRecommendationClient.RecommendationResponse>> getRecommendationsAsync(
            @RequestParam String bookTitle,
            @RequestParam(defaultValue = "5") Integer nRecommendations) {

        log.info("📚 Asenkron öneri endpoint çağrıldı: '{}' için {} öneri", bookTitle, nRecommendations);

        return bookRecommendationTestService.getRecommendationsAsync(bookTitle, nRecommendations)
                .map(ResponseEntity::ok);
    }

    /**
     * Model istatistikleri
     */
    @GetMapping("/model/stats")
    public ResponseEntity<BookRecommendationClient.ModelStats> getModelStats() {
        log.info("📊 Model istatistik endpoint çağrıldı");
        BookRecommendationClient.ModelStats stats = bookRecommendationTestService.getModelStatsSync();
        return ResponseEntity.ok(stats);
    }

    /**
     * Test raporu oluştur
     */
    @GetMapping("/report")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> generateTestReport() {
        log.info("📋 Test raporu endpoint çağrıldı");

        return bookRecommendationTestService.runAllTestsAsync()
                .thenApply(report -> ResponseEntity.ok(Map.of(
                        "success", true,
                        "report", report,
                        "timestamp", System.currentTimeMillis()
                )));
    }

    /**
     * Toplu kitap testi
     */
    @PostMapping("/bulk-test")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> bulkTestBooks(
            @RequestBody Map<String, Object> request) {

        @SuppressWarnings("unchecked")
        java.util.List<String> bookTitles = (java.util.List<String>) request.get("bookTitles");

        if (bookTitles == null || bookTitles.isEmpty()) {
            return CompletableFuture.completedFuture(
                    ResponseEntity.badRequest().body(Map.of(
                            "success", false,
                            "message", "bookTitles listesi gerekli"
                    ))
            );
        }

        log.info("📚 Toplu test endpoint çağrıldı: {} kitap", bookTitles.size());

        return bookRecommendationTestService.bulkTestBooks(bookTitles.toArray(new String[0]))
                .thenApply(results -> ResponseEntity.ok(Map.of(
                        "success", true,
                        "results", results,
                        "total_tested", bookTitles.size(),
                        "timestamp", System.currentTimeMillis()
                )));
    }

    /**
     * Hızlı test (önceden tanımlanmış kitaplar)
     */
    @GetMapping("/quick-test")
    public CompletableFuture<ResponseEntity<Map<String, Object>>> quickTest() {
        log.info("⚡ Hızlı test endpoint çağrıldı");

        String[] testBooks = {"körlük", "suç ve ceza", "anna karenina", "görmek"};

        return bookRecommendationTestService.bulkTestBooks(testBooks)
                .thenApply(results -> ResponseEntity.ok(Map.of(
                        "success", true,
                        "message", "Hızlı test tamamlandı",
                        "results", results,
                        "test_books", testBooks,
                        "timestamp", System.currentTimeMillis()
                )));
    }

    /**
     * API durumu
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getApiStatus() {
        log.info("🔍 API durum endpoint çağrıldı");

        try {
            BookRecommendationClient.HealthResponse health = bookRecommendationTestService.checkHealthSync();
            BookRecommendationClient.ModelStats stats = bookRecommendationTestService.getModelStatsSync();

            return ResponseEntity.ok(Map.of(
                    "api_status", "running",
                    "fastapi_health", health.getStatus(),
                    "model_loaded", health.getModelLoaded(),
                    "total_books", stats.getTotalBooks(),
                    "spring_boot_status", "healthy",
                    "timestamp", System.currentTimeMillis()
            ));

        } catch (Exception e) {
            return ResponseEntity.ok(Map.of(
                    "api_status", "running",
                    "fastapi_health", "unknown",
                    "spring_boot_status", "healthy",
                    "error", e.getMessage(),
                    "timestamp", System.currentTimeMillis()
            ));
        }
    }
}