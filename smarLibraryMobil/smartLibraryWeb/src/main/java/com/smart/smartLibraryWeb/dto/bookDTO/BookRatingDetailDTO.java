package com.smart.smartLibraryWeb.dto.bookDTO;


import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class BookRatingDetailDTO {
    private Long bookId;
    private String bookTitle;
    private Double averageRating;
    private Integer totalRatings;
    private List<String> comments;
}

