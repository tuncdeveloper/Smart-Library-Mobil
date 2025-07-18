package com.smart.smartLibraryWeb.mapper;

public interface BaseMapper<E,D> {
    D mapToDto(E entity);
    E mapToEntity(D dto);
}
