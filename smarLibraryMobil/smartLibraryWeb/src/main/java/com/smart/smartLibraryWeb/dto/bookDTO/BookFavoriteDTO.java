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
public class BookFavoriteDTO {
    private Long studentId;
    private Long bookId;
    private String title;
    private String author;
    private String category;
    private Integer publicationYear;


}

