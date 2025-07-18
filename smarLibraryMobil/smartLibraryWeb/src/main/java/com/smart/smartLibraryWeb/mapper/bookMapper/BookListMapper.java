package com.smart.smartLibraryWeb.mapper.bookMapper;


import com.smart.smartLibraryWeb.dto.bookDTO.BookListDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Book;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface BookListMapper extends BaseMapper<Book, BookListDTO> {
    @Override
    BookListDTO mapToDto(Book entity);

    @Override
    Book mapToEntity(BookListDTO dto);
}
