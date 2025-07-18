package com.smart.smartLibraryWeb.service.bookService;

import com.smart.smartLibraryWeb.dto.bookDTO.*;
import com.smart.smartLibraryWeb.exception.BadRequestException;
import com.smart.smartLibraryWeb.exception.ResourceNotFoundException;
import com.smart.smartLibraryWeb.mapper.bookMapper.*;
import com.smart.smartLibraryWeb.model.Book;
import com.smart.smartLibraryWeb.model.BookFavorite;
import com.smart.smartLibraryWeb.model.BookRating;
import com.smart.smartLibraryWeb.model.Student;
import com.smart.smartLibraryWeb.repository.BookFavoriteRepository;
import com.smart.smartLibraryWeb.repository.BookRatingRepository;
import com.smart.smartLibraryWeb.repository.BookRepository;
import com.smart.smartLibraryWeb.repository.StudentRepository;
import lombok.RequiredArgsConstructor;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class BookServiceImpl implements BookService {

    private final BookRepository bookRepository;
    private final BookRatingRepository bookRatingRepository;
    private final StudentRepository studentRepository;
    private final BookFavoriteRepository bookFavoriteRepository;
    private final BookFavoriteMapper bookFavoriteMapper;
    private final BookCreateMapper bookCreateMapper;
    private final BookListMapper bookListMapper;
    private final BookDetailMapper bookDetailMapper;
    private final BookRatingMapper bookRatingMapper;

    @Override
    @Transactional
    public void createBook(BookCreateDTO createDTO) {
        Book book = bookCreateMapper.mapToEntity(createDTO);
        bookRepository.save(book);
    }



    @Override
    @Transactional
    public void deleteBook(Long id) {
        if (!bookRepository.existsById(id)) {
            throw new RuntimeException("Kitap bulunamadı! ID: " + id);
        }
        bookRepository.deleteById(id);
    }

    @Override
    public List<BookListDTO> getAllBooks() {
        return bookRepository.findAll().stream()
                .map(bookListMapper::mapToDto)
                .collect(Collectors.toList());
    }

    @Override
    public BookDetailDTO getBookById(Long id) {
        Book book = bookRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Kitap bulunamadı! ID: " + id));

        try {
            return bookDetailMapper.mapToDto(book);
        } catch (Exception e) {
            Logger log = null;
            log.error("Mapper hatası: ", e);
            throw new RuntimeException("Kitap detayı oluşturulurken hata oluştu", e);
        }
    }

    @Override
    public List<BookListDTO> getBooksByCategory(String category) {
        return bookRepository.findByCategory(category).stream()
                .map(bookListMapper::mapToDto)
                .collect(Collectors.toList());
    }

    @Override
    public List<BookListDTO> searchBooksByTitle(String titleKeyword) {
        return bookRepository.findByTitleContainingIgnoreCase(titleKeyword).stream()
                .map(bookListMapper::mapToDto)
                .collect(Collectors.toList());
    }




    @Override
    @Transactional
    public BookRatingResponseDTO rateBook(BookRatingRequestDTO dto) {
        // Input validation
        if (dto.getRating() < 1 || dto.getRating() > 5) {
            throw new IllegalArgumentException("Puan 1 ile 5 arasında olmalı.");
        }

        // Student kontrolü
        Student student = studentRepository.findById(dto.getStudentId())
                .orElseThrow(() -> new IllegalArgumentException("Öğrenci bulunamadı! ID: " + dto.getStudentId()));

        // Book kontrolü
        Book book = bookRepository.findById(dto.getBookId())
                .orElseThrow(() -> new IllegalArgumentException("Kitap bulunamadı! ID: " + dto.getBookId()));

        // Mevcut puanlama var mı kontrol et
        Optional<BookRating> existingRating = bookRatingRepository.findByBookIdAndStudentId(dto.getBookId(), dto.getStudentId());

        BookRating bookRating;
        boolean isNewRating = existingRating.isEmpty();

        if (existingRating.isPresent()) {
            // Mevcut puanlamayı güncelle
            bookRating = existingRating.get();
            bookRating.setRating(dto.getRating());
            bookRating.setCreatedAt(LocalDateTime.now());
        } else {
            // Yeni puanlama oluştur
            bookRating = new BookRating();
            bookRating.setStudent(student);
            bookRating.setBook(book);
            bookRating.setRating(dto.getRating());
            bookRating.setCreatedAt(LocalDateTime.now());
        }

        BookRating savedRating = bookRatingRepository.save(bookRating);

        // ⭐ Sadece yeni rating ise total_ratings'i artır
        if (isNewRating) {
            Integer currentTotal = book.getTotalRatings() != null ? book.getTotalRatings() : 0;
            book.setTotalRatings(currentTotal + 1);
        }

        // Ortalamayı her zaman yeniden hesapla
        Double avgRating = bookRatingRepository.getAverageRatingByBookId(dto.getBookId());
        book.setAverageRating(avgRating != null ? Math.round(avgRating * 100.0) / 100.0 : 0.0);

        bookRepository.save(book);

        return bookRatingMapper.mapToResponseDto(savedRating);
    }



    @Override
    @Transactional
    public BookRatingResponseDTO commentBook(BookRatingRequestDTO dto) {
        if (dto.getComment() == null || dto.getComment().trim().isEmpty()) {
            throw new IllegalArgumentException("Yorum boş olamaz.");
        }

        Student student = studentRepository.findById(dto.getStudentId())
                .orElseThrow(() -> new IllegalArgumentException("Öğrenci bulunamadı! ID: " + dto.getStudentId()));

        Book book = bookRepository.findById(dto.getBookId())
                .orElseThrow(() -> new IllegalArgumentException("Kitap bulunamadı! ID: " + dto.getBookId()));

        Optional<BookRating> existingRatingOpt = bookRatingRepository.findByBookIdAndStudentId(dto.getBookId(), dto.getStudentId());

        BookRating bookRating;

        if (existingRatingOpt.isPresent()) {
            bookRating = existingRatingOpt.get();
            bookRating.setComment(dto.getComment());
            if (dto.getRating() != 0) {
                bookRating.setRating(dto.getRating());
            }
            bookRating.setCreatedAt(LocalDateTime.now());
        } else {
            bookRating = new BookRating();
            bookRating.setBook(book);
            bookRating.setStudent(student);
            bookRating.setComment(dto.getComment());
            // DTO'daki rating değeri varsa kullan, yoksa 0 koy
            bookRating.setRating(dto.getRating() != 0 ? dto.getRating() : 0);
            bookRating.setCreatedAt(LocalDateTime.now());
        }

        BookRating saved = bookRatingRepository.save(bookRating);

        // Eğer yeni yorum veya rating eklenmişse, ortalamayı güncelle
        Double avgRating = bookRatingRepository.getAverageRatingByBookId(dto.getBookId());
        book.setAverageRating(avgRating != null ? Math.round(avgRating * 100.0) / 100.0 : 0.0);
        bookRepository.save(book);

        return bookRatingMapper.mapToResponseDto(saved);
    }



    @Override
    public double getAverageRating(Long bookId) {
        // Kitap var mı kontrol et
        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new IllegalArgumentException("Kitap bulunamadı! ID: " + bookId));

        // Önce Book entity'sinden kontrol et
        if (book.getAverageRating() != null) {
            return book.getAverageRating();
        }

        // Yoksa veritabanından hesapla
        Double avg = bookRatingRepository.getAverageRatingByBookId(bookId);
        return avg != null ? avg : 0.0;
    }

    @Override
    public Integer getUserRatingForBook(Long bookId, Long userId) {
        // Kitap ve kullanıcı var mı kontrol et
        if (!bookRepository.existsById(bookId)) {
            throw new IllegalArgumentException("Kitap bulunamadı! ID: " + bookId);
        }

        if (!studentRepository.existsById(userId)) {
            throw new IllegalArgumentException("Kullanıcı bulunamadı! ID: " + userId);
        }

        return bookRatingRepository.findByBookIdAndStudentId(bookId, userId)
                .map(BookRating::getRating)
                .orElse(null);
    }

    @Override
    public Integer getTotalRatingsCount(Long bookId) {
        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new IllegalArgumentException("Kitap bulunamadı! ID: " + bookId));

        // Önce Book entity'sinden kontrol et
        if (book.getTotalRatings() != null) {
            return book.getTotalRatings();
        }

        // Yoksa veritabanından say
        Integer count = bookRatingRepository.countByBookId(bookId);

        // Entity'yi güncelle
        book.setTotalRatings(count);
        bookRepository.save(book);

        return count;
    }

    @Override
    public BookRatingDetailDTO getBookRatingDetail(Long bookId) {
        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new IllegalArgumentException("Kitap bulunamadı! ID: " + bookId));

        BookRatingDetailDTO detail = new BookRatingDetailDTO();
        detail.setBookId(bookId);
        detail.setBookTitle(book.getTitle());

        // Entity'de rating bilgileri varsa kullan
        if (book.getAverageRating() != null && book.getTotalRatings() != null) {
            detail.setAverageRating(book.getAverageRating());
            detail.setTotalRatings(book.getTotalRatings());
        } else {
            // Yoksa hesapla ve güncelle
            Double avg = bookRatingRepository.getAverageRatingByBookId(bookId);
            Integer count = bookRatingRepository.countByBookId(bookId);
            String reliability = count < 5 ? "Az" : (count < 20 ? "Orta" : "Yüksek");

            detail.setAverageRating(avg != null ? avg : 0.0);
            detail.setTotalRatings(count);

            // Book entity'sini güncelle
            book.setAverageRating(avg != null ? avg : 0.0);
            book.setTotalRatings(count);
            bookRepository.save(book);
        }

        return detail;
    }

    @Override
    public Optional<Book> findById(Long id) {
        return bookRepository.findById(id);
    }


    public List<BookRatingResponseDTO> getAllCommentsForBook(Long bookId) {
        return bookRatingRepository.findAllRatingDTOsByBookId(bookId);
    }

    @Override
    @Transactional
    public BookFavoriteDTO addFavoriteBook(Long studentId, Long bookId) {
        Student student = studentRepository.findById(studentId)
                .orElseThrow(() -> new ResourceNotFoundException("Öğrenci bulunamadı"));

        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new ResourceNotFoundException("Kitap bulunamadı"));

        boolean exists = bookFavoriteRepository.findByStudentAndBook(student, book).isPresent();
        if (exists) {
            throw new BadRequestException("Kitap zaten favorilere eklenmiş");
        }

        BookFavorite favorite = new BookFavorite();
        favorite.setStudent(student);
        favorite.setBook(book);

        BookFavorite savedFavorite = bookFavoriteRepository.save(favorite);

        return bookFavoriteMapper.mapToDto(savedFavorite);
    }

    @Override
    @Transactional
    public BookFavoriteDTO removeFavoriteBook(Long studentId, Long bookId) {
        Student student = studentRepository.findById(studentId)
                .orElseThrow(() -> new ResourceNotFoundException("Öğrenci bulunamadı"));

        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new ResourceNotFoundException("Kitap bulunamadı"));

        BookFavorite favorite = bookFavoriteRepository.findByStudentAndBook(student, book)
                .orElseThrow(() -> new ResourceNotFoundException("Favori kitap bulunamadı"));

        bookFavoriteRepository.delete(favorite);

        return bookFavoriteMapper.mapToDto(favorite);
    }


    @Override
    public List<BookFavoriteDTO> getFavoriteBooksByStudentId(Long studentId) {
        List<BookFavorite> favorites = bookFavoriteRepository.findByStudentId(studentId);
        return favorites.stream()
                .map(bookFavoriteMapper::mapToDto)
                .collect(Collectors.toList());
    }


}