package com.smart.smartLibraryWeb.dto.bookDTO;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class BookRatingResponseDTO {
    private Long id; // rating ID
    private Long studentId;
    private Long bookId;
    private String comment;
    private int rating;
    private String bookTitle;
    private String studentUsername;
    private LocalDateTime createdAt;
}


