package com.smart.smartLibraryWeb.repository;

import com.smart.smartLibraryWeb.model.Book;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;


public interface BookRepository extends JpaRepository<Book, Long> {


    List<Book> findByCategory(String category);

    List<Book> findByTitleContainingIgnoreCase(String titleKeyword);


    // Sayfalı olarak kategoriye göre

}



