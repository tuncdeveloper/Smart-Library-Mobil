package com.smart.smartLibraryWeb.controller;


import com.smart.smartLibraryWeb.dto.studentDTO.*;
import com.smart.smartLibraryWeb.service.studentService.StudentService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/students")
@RequiredArgsConstructor
public class StudentController {

    private final StudentService studentService;



    // Tüm öğrencileri listele
    @GetMapping
    public ResponseEntity<List<StudentListDTO>> getAllStudents() {
        List<StudentListDTO> students = studentService.getAllStudents();
        return ResponseEntity.ok(students);
    }

    // ID'ye göre öğrenci getir
    @GetMapping("/{id}")
    public ResponseEntity<StudentListDTO> getStudentById(@PathVariable Long id) {
        try {
            StudentListDTO student = studentService.getStudentById(id);
            return ResponseEntity.ok(student);
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
    }

    // Öğrenci güncelleme
    @PutMapping("/{id}")
    public ResponseEntity<StudentUpdateDTO> updateStudent(@PathVariable Long id, @RequestBody StudentUpdateDTO updateDTO) {
        try {
            updateDTO.setId(id); // DTO'nun ID'sini URL'den gelen ID ile eşle
            StudentUpdateDTO updatedStudent = studentService.updateStudent(updateDTO);
            return ResponseEntity.ok(updatedStudent); // DTO dönülüyor
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).build();
        }
    }


    // Öğrenci silme
    @DeleteMapping("/{id}")
    public ResponseEntity<String> deleteStudent(@PathVariable Long id) {
        try {
            studentService.deleteStudent(id);
            return ResponseEntity.ok("Silme başarılı");
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());
        }
    }
}
