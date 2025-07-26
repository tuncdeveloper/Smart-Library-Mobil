package com.smart.smartLibraryWeb.test;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import static org.mockito.ArgumentMatchers.any;

@SpringBootTest
@AutoConfigureWebTestClient
public class RecommendationControllerTest {

    @Autowired
    private WebTestClient webTestClient;

    @MockBean
    private BookRecommendationService recommendationService;

    @Test
    void shouldReturnPrediction() {
        PredictionResponse mockResponse = new PredictionResponse();
        mockResponse.setPrediction(3);
        mockResponse.setConfidence(0.95);

        Mockito.when(recommendationService.predictBookRecommendation(any()))
                .thenReturn(Mono.just(mockResponse));

        webTestClient.post()
                .uri("/api/recommendations")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue("{\"features\":[1.2,3.4,5.6]}")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.prediction").isEqualTo(3)
                .jsonPath("$.confidence").isEqualTo(0.95);
    }
}
