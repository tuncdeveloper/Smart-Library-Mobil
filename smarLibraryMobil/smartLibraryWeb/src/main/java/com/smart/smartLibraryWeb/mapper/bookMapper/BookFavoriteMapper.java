package com.smart.smartLibraryWeb.mapper.bookMapper;

import com.smart.smartLibraryWeb.dto.bookDTO.BookFavoriteDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.BookFavorite;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface BookFavoriteMapper extends BaseMapper<BookFavorite, BookFavoriteDTO> {

    @Override
    @Mapping(source = "student.id", target = "studentId")
    @Mapping(source = "book.id", target = "bookId")
    @Mapping(source = "book.title", target = "title")
    @Mapping(source = "book.author", target = "author")
    @Mapping(source = "book.category", target = "category")
    @Mapping(source = "book.publicationYear", target = "publicationYear")
    BookFavoriteDTO mapToDto(BookFavorite entity);

    @Override
    BookFavorite mapToEntity(BookFavoriteDTO dto);
}

