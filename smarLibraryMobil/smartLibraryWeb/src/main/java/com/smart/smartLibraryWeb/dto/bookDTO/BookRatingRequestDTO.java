package com.smart.smartLibraryWeb.dto.bookDTO;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class BookRatingRequestDTO {
    private Long studentId;
    private Long bookId;
    private String comment;
    private int rating; // 1–5
}

