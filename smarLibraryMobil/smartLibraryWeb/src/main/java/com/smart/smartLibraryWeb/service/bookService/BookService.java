package com.smart.smartLibraryWeb.service.bookService;

import com.smart.smartLibraryWeb.dto.bookDTO.*;
import com.smart.smartLibraryWeb.model.Book;

import java.util.List;
import java.util.Optional;

public interface BookService {

    // Kitap ekleme
    void createBook(BookCreateDTO createDTO);

    // Kitap güncelleme

    // Kitap silme
    void deleteBook(Long id);

    // Tüm kitapları listeleme
    List<BookListDTO> getAllBooks();

    // ID ile kitap detayını getirme
    BookDetailDTO getBookById(Long id);


    BookFavoriteDTO addFavoriteBook(Long studentId, Long bookId);
    BookFavoriteDTO removeFavoriteBook(Long studentId, Long bookId);


    // Öğrencinin favori kitaplarını listeleme
    List<BookFavoriteDTO> getFavoriteBooksByStudentId(Long studentId);
    // Kategoriye göre kitapları listeleme
    List<BookListDTO> getBooksByCategory(String category);

    // Başlığa göre arama
    List<BookListDTO> searchBooksByTitle(String titleKeyword);

    // Mevcut (aktif) kitapları getir

    // Basit kitap listesi (id ve title gibi)

    // ⭐ Rating metodları
    BookRatingResponseDTO rateBook(BookRatingRequestDTO dto);

    // Kitabın ortalama puanını getir
    double getAverageRating(Long bookId);

    // Kullanıcının verdiği puanı getir
    Integer getUserRatingForBook(Long bookId, Long userId);

    // Kitabın toplam puan sayısını getir
    Integer getTotalRatingsCount(Long bookId);

    // Kitabın detaylı rating bilgilerini getir
    BookRatingDetailDTO getBookRatingDetail(Long bookId);

    Optional<Book> findById(Long id);

     BookRatingResponseDTO commentBook(BookRatingRequestDTO dto) ;

     List<BookRatingResponseDTO> getAllCommentsForBook(Long bookId);

    }