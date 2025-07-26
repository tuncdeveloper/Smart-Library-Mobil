package com.smart.smartLibraryWeb.config;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.List;
import java.util.Map;

@Slf4j
@Component
public class BookRecommendationClient {

    private final WebClient webClient;
    private final ObjectMapper objectMapper;

    public BookRecommendationClient() {
        this.webClient = WebClient.builder()
                .baseUrl("http://localhost:8000")
                .defaultHeader("Content-Type", "application/json")
                .build();
        this.objectMapper = new ObjectMapper();
    }

    // DTO Classes
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RecommendationRequest {
        @JsonProperty("book_title")
        private String bookTitle;

        @JsonProperty("n_recommendations")
        private Integer nRecommendations;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class BookRecommendation {
        private Integer rank;
        private String title;
        private String author;
        private String category;
        private String publisher;

        @JsonProperty("publication_year")
        private Integer publicationYear;

        @JsonProperty("average_rating")
        private Double averageRating;

        @JsonProperty("total_ratings")
        private Integer totalRatings;

        @JsonProperty("page_count")
        private Integer pageCount;

        private String description;

        @JsonProperty("similarity_score")
        private Double similarityScore;

        @JsonProperty("confidence_score")
        private Double confidenceScore;

        @JsonProperty("diversity_score")
        private Double diversityScore;

        @JsonProperty("final_score")
        private Double finalScore;

        @JsonProperty("confidence_details")
        private Map<String, Double> confidenceDetails;

        @JsonProperty("recommendation_reasons")
        private List<String> recommendationReasons;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TargetBook {
        private String title;
        private String author;
        private String category;
        private String publisher;

        @JsonProperty("publication_year")
        private Integer publicationYear;

        @JsonProperty("average_rating")
        private Double averageRating;

        @JsonProperty("total_ratings")
        private Integer totalRatings;

        private String description;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class RecommendationResponse {
        private Boolean success;
        private String message;

        @JsonProperty("target_book")
        private TargetBook targetBook;

        private List<BookRecommendation> recommendations;

        @JsonProperty("total_recommendations")
        private Integer totalRecommendations;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class HealthResponse {
        private String status;

        @JsonProperty("model_loaded")
        private Boolean modelLoaded;

        private String message;
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ModelStats {
        @JsonProperty("total_books")
        private Integer totalBooks;

        @JsonProperty("unique_authors")
        private Integer uniqueAuthors;

        @JsonProperty("unique_categories")
        private Integer uniqueCategories;

        @JsonProperty("model_features")
        private Integer modelFeatures;

        private Map<String, Object> hyperparameters;

        @JsonProperty("confidence_threshold")
        private Double confidenceThreshold;
    }

    // API Methods

    /**
     * Health check endpoint'ini test eder
     */
    public Mono<HealthResponse> checkHealth() {
        log.info("🏥 Health check yapılıyor...");

        return webClient.get()
                .uri("/health")
                .retrieve()
                .bodyToMono(HealthResponse.class)
                .timeout(Duration.ofSeconds(10))
                .doOnSuccess(response -> log.info("✅ Health check başarılı: {}", response))
                .doOnError(error -> log.error("❌ Health check hatası: {}", error.getMessage()));
    }

    /**
     * GET metodu ile kitap önerisi alır
     */
    public Mono<RecommendationResponse> getRecommendationsGet(String bookTitle, Integer nRecommendations) {
        log.info("📚 GET ile '{}' için {} öneri isteniyor...", bookTitle, nRecommendations);

        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/recommendations/{bookTitle}")
                        .queryParam("n_recommendations", nRecommendations)
                        .build(bookTitle))
                .retrieve()
                .bodyToMono(RecommendationResponse.class)
                .timeout(Duration.ofSeconds(30))
                .doOnSuccess(response -> {
                    log.info("✅ GET Öneriler alındı: {} öneri", response.getTotalRecommendations());
                    if (response.getRecommendations() != null) {
                        response.getRecommendations().forEach(rec ->
                                log.info("  {}. {} - {} (Skor: {:.4f})",
                                        rec.getRank(), rec.getTitle(), rec.getAuthor(), rec.getFinalScore())
                        );
                    }
                })
                .doOnError(error -> log.error("❌ GET Öneri hatası: {}", error.getMessage()));
    }

    /**
     * Model istatistiklerini alır
     */
    public Mono<ModelStats> getModelStats() {
        log.info("📊 Model istatistikleri alınıyor...");

        return webClient.get()
                .uri("/model/stats")
                .retrieve()
                .bodyToMono(ModelStats.class)
                .timeout(Duration.ofSeconds(10))
                .doOnSuccess(stats -> log.info("✅ Model istatistikleri: {} kitap, {} yazar, {} kategori",
                        stats.getTotalBooks(), stats.getUniqueAuthors(), stats.getUniqueCategories()))
                .doOnError(error -> log.error("❌ Model istatistik hatası: {}", error.getMessage()));
    }

    /**
     * Kategorileri alır
     */
    public Mono<Map> getCategories() {
        log.info("📂 Kategoriler alınıyor...");

        return webClient.get()
                .uri("/categories")
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(10))
                .doOnSuccess(response -> {
                    @SuppressWarnings("unchecked")
                    List<String> categories = (List<String>) response.get("categories");
                    log.info("✅ {} kategori alındı", categories != null ? categories.size() : 0);
                })
                .doOnError(error -> log.error("❌ Kategori alma hatası: {}", error.getMessage()));
    }

    /**
     * Yazarları alır
     */
    public Mono<Map> getAuthors() {
        log.info("👤 Yazarlar alınıyor...");

        return webClient.get()
                .uri("/authors")
                .retrieve()
                .bodyToMono(Map.class)
                .timeout(Duration.ofSeconds(10))
                .doOnSuccess(response -> {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> authors = (List<Map<String, Object>>) response.get("authors");
                    log.info("✅ {} yazar alındı", authors != null ? authors.size() : 0);
                })
                .doOnError(error -> log.error("❌ Yazar alma hatası: {}", error.getMessage()));
    }

    /**
     * Tüm endpoint'leri test eder
     */
    public Mono<Void> runAllTests() {
        log.info("🚀 Tüm API testleri başlıyor...");

        return checkHealth()
                .flatMap(health -> {
                    if (!health.getModelLoaded()) {
                        log.warn("⚠️ Model yüklenmemiş, testler başarısız olabilir");
                    }
                    return getModelStats();
                })
                .flatMap(stats -> {
                    log.info("📊 Model hazır: {} kitap mevcut", stats.getTotalBooks());
                    return getCategories();
                })
                .flatMap(categories -> getAuthors())
                .flatMap(authors -> {
                    // Test kitapları
                    String[] testBooks = {"körlük", "suç ve ceza", "anna karenina", "görmek"};

                    return Mono.fromRunnable(() -> {
                        for (String bookTitle : testBooks) {
                            getRecommendationsGet(bookTitle, 3)
                                    .subscribe(
                                            response -> log.info("✅ '{}' için {} öneri alındı",
                                                    bookTitle, response.getTotalRecommendations()),
                                            error -> log.error("❌ '{}' için hata: {}", bookTitle, error.getMessage())
                                    );
                        }
                    });
                })
                .doOnSuccess(v -> log.info("🎉 Tüm testler tamamlandı!"))
                .doOnError(error -> log.error("💥 Test sırasında hata: {}", error.getMessage()))
                .then();
    }

    /**
     * Detaylı test raporu oluşturur
     */
    public Mono<String> generateTestReport() {
        log.info("📋 Test raporu oluşturuluyor...");

        StringBuilder report = new StringBuilder();
        report.append("=".repeat(80)).append("\n");
        report.append("📚 BOOK RECOMMENDATION API TEST RAPORU\n");
        report.append("=".repeat(80)).append("\n");

        return checkHealth()
                .flatMap(health -> {
                    report.append(String.format("🏥 Health Status: %s (Model: %s)\n",
                            health.getStatus(), health.getModelLoaded() ? "✅" : "❌"));
                    return getModelStats();
                })
                .flatMap(stats -> {
                    report.append(String.format("📊 Model Stats:\n"));
                    report.append(String.format("   - Toplam Kitap: %d\n", stats.getTotalBooks()));
                    report.append(String.format("   - Yazar Sayısı: %d\n", stats.getUniqueAuthors()));
                    report.append(String.format("   - Kategori Sayısı: %d\n", stats.getUniqueCategories()));
                    report.append(String.format("   - Özellik Boyutu: %d\n", stats.getModelFeatures()));
                    report.append(String.format("   - Güven Eşiği: %.2f\n\n", stats.getConfidenceThreshold()));

                    return getRecommendationsGet("körlük", 5);
                })
                .flatMap(recommendations -> {
                    report.append("🔍 Test Önerisi: 'körlük'\n");
                    report.append(String.format("   Hedef Kitap: %s - %s\n",
                            recommendations.getTargetBook().getTitle(),
                            recommendations.getTargetBook().getAuthor()));
                    report.append(String.format("   Öneri Sayısı: %d\n\n", recommendations.getTotalRecommendations()));

                    report.append("📋 Öneriler:\n");
                    if (recommendations.getRecommendations() != null) {
                        for (BookRecommendation rec : recommendations.getRecommendations()) {
                            report.append(String.format("   %d. %s - %s\n",
                                    rec.getRank(), rec.getTitle(), rec.getAuthor()));
                            report.append(String.format("      Kategori: %s | Final Skor: %.4f\n",
                                    rec.getCategory(), rec.getFinalScore()));
                            report.append(String.format("      Nedenler: %s\n",
                                    String.join(", ", rec.getRecommendationReasons())));
                        }
                    }

                    report.append("\n").append("=".repeat(80)).append("\n");
                    report.append("✅ Test Raporu Tamamlandı!\n");
                    report.append("=".repeat(80));

                    return Mono.just(report.toString());
                })
                .doOnSuccess(reportStr -> log.info("📋 Test raporu hazır:\n{}", reportStr))
                .doOnError(error -> log.error("❌ Test raporu hatası: {}", error.getMessage()));
    }
}