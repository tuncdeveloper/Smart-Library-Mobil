package com.smart.smartLibraryWeb.model;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Entity
@Table(name = "favorite_books")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class BookFavorite {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)  // Bir favori bir öğrenciye ait
    @JoinColumn(name = "student_id", nullable = false)
    private Student student;

    @ManyToOne(fetch = FetchType.LAZY)  // Bir favori bir kitaba ait
    @JoinColumn(name = "book_id", nullable = false)
    private Book book;


}

