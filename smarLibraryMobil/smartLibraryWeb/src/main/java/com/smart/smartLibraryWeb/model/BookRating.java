package com.smart.smartLibraryWeb.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@NoArgsConstructor
public class BookRating {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private int rating; // 1–5 arasında

    @ManyToOne
    private Book book;

    @ManyToOne
    private Student student;

    private LocalDateTime createdAt;

    @Column(length = 1000) // Uzun yorumlara izin vermek için
    private String comment;

}
