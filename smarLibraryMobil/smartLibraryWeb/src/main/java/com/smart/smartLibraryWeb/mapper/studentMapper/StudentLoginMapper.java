package com.smart.smartLibraryWeb.mapper.studentMapper;


import com.smart.smartLibraryWeb.dto.AuthRequest;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Student;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface StudentLoginMapper extends BaseMapper<Student, AuthRequest> {


    @Override
    AuthRequest mapToDto(Student entity);

    @Override
    Student mapToEntity(AuthRequest dto);



}
