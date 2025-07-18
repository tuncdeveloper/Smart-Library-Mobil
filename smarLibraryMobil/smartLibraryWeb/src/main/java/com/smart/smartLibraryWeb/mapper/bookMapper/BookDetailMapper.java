package com.smart.smartLibraryWeb.mapper.bookMapper;


import com.smart.smartLibraryWeb.dto.bookDTO.BookDetailDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Book;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface BookDetailMapper extends BaseMapper<Book, BookDetailDTO> {
    @Override
    BookDetailDTO mapToDto(Book entity);

    @Override
    Book mapToEntity(BookDetailDTO dto);
}
