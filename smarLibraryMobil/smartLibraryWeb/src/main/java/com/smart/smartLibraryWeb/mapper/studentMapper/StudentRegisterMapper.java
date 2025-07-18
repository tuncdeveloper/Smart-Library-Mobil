package com.smart.smartLibraryWeb.mapper.studentMapper;


import com.smart.smartLibraryWeb.dto.studentDTO.StudentRegisterDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Student;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface StudentRegisterMapper extends BaseMapper<Student, StudentRegisterDTO> {

    @Override
    StudentRegisterDTO mapToDto(Student entity);

    @Override
    Student mapToEntity(StudentRegisterDTO dto);
}
