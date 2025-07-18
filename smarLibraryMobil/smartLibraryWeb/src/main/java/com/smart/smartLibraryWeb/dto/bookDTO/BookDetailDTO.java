package com.smart.smartLibraryWeb.dto.bookDTO;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BookDetailDTO {
    private Long id;
    private String title;
    private String author;
    private String publisher;
    private Integer publicationYear;
    private String category;
    private String language;
    private Integer pageCount;
    private String description;
    private LocalDateTime addedDate;
    private Double averageRating;
    private Integer totalRatings;
}
