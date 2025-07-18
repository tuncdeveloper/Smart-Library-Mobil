package com.smart.smartLibraryWeb.mapper.bookMapper;

import com.smart.smartLibraryWeb.dto.bookDTO.BookRatingRequestDTO;
import com.smart.smartLibraryWeb.dto.bookDTO.BookRatingResponseDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.BookRating;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE, componentModel = "spring")
public interface BookRatingMapper extends BaseMapper<BookRating, BookRatingRequestDTO> {

    @Override
    BookRatingRequestDTO mapToDto(BookRating entity);

    @Override
    BookRating mapToEntity(BookRatingRequestDTO dto);

    // Response DTO için ek mapping method
    @Mapping(source = "book.id", target = "bookId")
    @Mapping(source = "student.id", target = "studentId")
    @Mapping(source = "book.title", target = "bookTitle")                 // burayı ekledik
    @Mapping(source = "student.username", target = "studentUsername")     // burayı ekledik
    BookRatingResponseDTO mapToResponseDto(BookRating entity);
}
