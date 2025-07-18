package com.smart.smartLibraryWeb.mapper.studentMapper;

import com.smart.smartLibraryWeb.dto.AuthResponse;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Student;

public interface StudentLoginResponseMapper extends BaseMapper<Student, AuthResponse> {

    @Override
    AuthResponse mapToDto(Student entity);

    @Override
    Student mapToEntity(AuthResponse dto);
}
