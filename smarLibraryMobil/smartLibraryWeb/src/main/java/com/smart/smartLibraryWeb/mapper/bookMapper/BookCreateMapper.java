package com.smart.smartLibraryWeb.mapper.bookMapper;


import com.smart.smartLibraryWeb.dto.bookDTO.BookCreateDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Book;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface BookCreateMapper extends BaseMapper<Book, BookCreateDTO> {
    @Override
    BookCreateDTO mapToDto(Book entity);

    @Override
    Book mapToEntity(BookCreateDTO dto);
}
