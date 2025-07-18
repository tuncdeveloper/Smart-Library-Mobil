package com.smart.smartLibraryWeb.service.studentService;

import com.smart.smartLibraryWeb.dto.studentDTO.*;

import java.util.List;

public interface StudentService {



    List<StudentListDTO> getAllStudents();

    StudentListDTO getStudentById(Long id);

    void deleteStudent(Long id);

    StudentUpdateDTO updateStudent(StudentUpdateDTO updateDto);
}

