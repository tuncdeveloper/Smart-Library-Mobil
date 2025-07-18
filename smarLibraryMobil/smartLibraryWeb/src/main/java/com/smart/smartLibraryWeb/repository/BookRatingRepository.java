package com.smart.smartLibraryWeb.repository;

import com.smart.smartLibraryWeb.dto.bookDTO.BookRatingResponseDTO;
import com.smart.smartLibraryWeb.model.BookRating;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface BookRatingRepository extends JpaRepository<BookRating, Long> {
    Optional<BookRating> findByBookIdAndStudentId(Long bookId, Long userId);

    List<BookRating> findAllByBookId(Long bookId);

    @Query("SELECT AVG(br.rating) FROM BookRating br WHERE br.book.id = :bookId")
    Double getAverageRatingByBookId(@Param("bookId") Long bookId);

    Integer countByBookId(Long bookId);

    @Query("SELECT new com.smart.smartLibraryWeb.dto.bookDTO.BookRatingResponseDTO( " +
            "br.id, s.id, b.id, br.comment, br.rating, b.title, s.username, br.createdAt) " +
            "FROM BookRating br " +
            "JOIN br.book b " +
            "JOIN br.student s " +
            "WHERE b.id = :bookId")
    List<BookRatingResponseDTO> findAllRatingDTOsByBookId(@Param("bookId") Long bookId);





}

