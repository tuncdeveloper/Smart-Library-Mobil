package com.smart.smartLibraryWeb.mapper.studentMapper;

import com.smart.smartLibraryWeb.dto.studentDTO.StudentUpdateDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Student;
import org.mapstruct.Mapper;
import org.mapstruct.MappingTarget;
import org.mapstruct.ReportingPolicy;


@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface StudentUpdateMapper extends BaseMapper<Student, StudentUpdateDTO> {

    @Override
    StudentUpdateDTO mapToDto(Student entity);

    @Override
    Student mapToEntity(StudentUpdateDTO dto);

    void updateStudentFromDto(StudentUpdateDTO dto, @MappingTarget Student student);

}
