package com.smart.smartLibraryWeb.repository;

import com.smart.smartLibraryWeb.model.Book;
import com.smart.smartLibraryWeb.model.BookFavorite;
import com.smart.smartLibraryWeb.model.Student;
import org.hibernate.sql.ast.tree.expression.JdbcParameter;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface BookFavoriteRepository extends JpaRepository<BookFavorite, Long> {
    Optional<BookFavorite> findByStudentAndBook(Student student, Book book);
    List<BookFavorite> findByStudentId(Long studentId);
}
