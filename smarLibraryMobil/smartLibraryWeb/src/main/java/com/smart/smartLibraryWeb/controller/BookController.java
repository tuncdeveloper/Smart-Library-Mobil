package com.smart.smartLibraryWeb.controller;

import com.smart.smartLibraryWeb.dto.bookDTO.*;
import com.smart.smartLibraryWeb.model.Book;
import com.smart.smartLibraryWeb.service.bookService.BookService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/books")
@RequiredArgsConstructor
public class BookController {

    private final BookService bookService;

    // Yeni kitap oluşturma
    @PostMapping
    public ResponseEntity<Void> createBook(@RequestBody BookCreateDTO createDTO) {
        bookService.createBook(createDTO);
        return new ResponseEntity<>(HttpStatus.CREATED);
    }




    // Kitap silme
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteBook(@PathVariable Long id) {
        bookService.deleteBook(id);
        return new ResponseEntity<>(HttpStatus.NO_CONTENT);
    }

    // Tüm kitapları listele
    @GetMapping
    public ResponseEntity<List<BookListDTO>> getAllBooks() {
        return ResponseEntity.ok(bookService.getAllBooks());
    }


    // ISBN'e göre kitap getir

    //id ye göre kitap getir
    @GetMapping("/id/{id}")
    public ResponseEntity<BookDetailDTO> getBookById(@PathVariable Long id) {
        return ResponseEntity.ok(bookService.getBookById(id));
    }

    // Kategoriye göre kitapları getir
    @GetMapping("/category/{category}")
    public ResponseEntity<List<BookListDTO>> getBooksByCategory(@PathVariable String category) {
        return ResponseEntity.ok(bookService.getBooksByCategory(category));
    }

    // Başlığa göre kitap arama
    @GetMapping("/search")
    public ResponseEntity<List<BookListDTO>> searchBooksByTitle(@RequestParam String title) {
        return ResponseEntity.ok(bookService.searchBooksByTitle(title));
    }


    @GetMapping("/list/{studentId}")
    public ResponseEntity<List<BookFavoriteDTO>> getFavorites(@PathVariable Long studentId) {
        List<BookFavoriteDTO> favorites = bookService.getFavoriteBooksByStudentId(studentId);
        return ResponseEntity.ok(favorites);
    }

    @PostMapping("/add")
    public ResponseEntity<BookFavoriteDTO> addFavorite(
            @RequestParam Long studentId,
            @RequestParam Long bookId) {
        BookFavoriteDTO favoriteDto = bookService.addFavoriteBook(studentId, bookId);
        return ResponseEntity.ok(favoriteDto);
    }

    @DeleteMapping("/remove")
    public ResponseEntity<BookFavoriteDTO> removeFavorite(
            @RequestParam Long studentId,
            @RequestParam Long bookId) {
        BookFavoriteDTO favoriteDto = bookService.removeFavoriteBook(studentId, bookId);
        return ResponseEntity.ok(favoriteDto);
    }



    // ⭐ Kitaba puan verme
    @PostMapping("/rate")
    public ResponseEntity<BookRatingResponseDTO> rateBook(@RequestBody BookRatingRequestDTO dto) {
        BookRatingResponseDTO response = bookService.rateBook(dto);
        return ResponseEntity.ok(response);
    }

    // ⭐ Ortalama puanı getir
    @GetMapping("/{bookId}/average-rating")
    public ResponseEntity<Double> getAverageRating(@PathVariable Long bookId) {
        double average = bookService.getAverageRating(bookId);
        return ResponseEntity.ok(average);
    }

    // ⭐ Belirli kullanıcının bu kitap için verdiği puan
    @GetMapping("/{bookId}/user-rating")
    public ResponseEntity<Integer> getUserRating(
            @PathVariable Long bookId,
            @RequestParam Long userId
    ) {
        Integer rating = bookService.getUserRatingForBook(bookId, userId);
        return ResponseEntity.ok(rating);
    }


    // ⭐ Kitabın toplam puan sayısını getir
    @GetMapping("/{bookId}/total-ratings")
    public ResponseEntity<Integer> getTotalRatingsCount(@PathVariable Long bookId) {
        Integer count = bookService.getTotalRatingsCount(bookId);
        return ResponseEntity.ok(count);
    }

    // ⭐ Kitabın detaylı rating bilgilerini getir
    @GetMapping("/{bookId}/rating-detail")
    public ResponseEntity<BookRatingDetailDTO> getBookRatingDetail(@PathVariable Long bookId) {
        BookRatingDetailDTO detail = bookService.getBookRatingDetail(bookId);
        return ResponseEntity.ok(detail);
    }

    // ⭐ Kitaba yorum yapma
    @PostMapping("/comment")
    public ResponseEntity<BookRatingResponseDTO> commentBook(@RequestBody BookRatingRequestDTO dto) {
        BookRatingResponseDTO response = bookService.commentBook(dto);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{bookId}/comments")
    public ResponseEntity<List<BookRatingResponseDTO>> getComments(@PathVariable Long bookId) {
        List<BookRatingResponseDTO> comments = bookService.getAllCommentsForBook(bookId);
        return ResponseEntity.ok(comments);
    }

}