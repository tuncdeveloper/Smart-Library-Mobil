package com.smart.smartLibraryWeb.service.studentService;

import com.smart.smartLibraryWeb.dto.studentDTO.*;
import com.smart.smartLibraryWeb.exception.BadRequestException;
import com.smart.smartLibraryWeb.exception.ResourceNotFoundException;
import com.smart.smartLibraryWeb.mapper.studentMapper.StudentListMapper;
import com.smart.smartLibraryWeb.mapper.studentMapper.StudentRegisterMapper;
import com.smart.smartLibraryWeb.mapper.studentMapper.StudentUpdateMapper;
import com.smart.smartLibraryWeb.model.Student;
import com.smart.smartLibraryWeb.repository.StudentRepository;
import jakarta.persistence.EntityNotFoundException;
import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class StudentServiceImpl implements StudentService {

    private final StudentRepository studentRepository;
    private final StudentListMapper studentListMapper;
    private final StudentUpdateMapper studentUpdateMapper;
    private final PasswordEncoder passwordEncoder; // PasswordEncoder eklendi

    @Override
    public List<StudentListDTO> getAllStudents() {
        return studentRepository.findAll().stream()
                .map(studentListMapper::mapToDto)
                .collect(Collectors.toList());
    }

    @Override
    public StudentListDTO getStudentById(Long id) {
        Student student = studentRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Öğrenci bulunamadı"));
        return studentListMapper.mapToDto(student);
    }

    @Override
    public void deleteStudent(Long id) {
        if (!studentRepository.existsById(id)) {
            throw new ResourceNotFoundException("Silinecek öğrenci bulunamadı");
        }
        studentRepository.deleteById(id);
    }

    @Override
    @Transactional
    public StudentUpdateDTO updateStudent(StudentUpdateDTO dto) {
        log.info("updateStudent called with DTO: {}", dto);

        Student existingStudent = studentRepository.findById(dto.getId())
                .orElseThrow(() -> new EntityNotFoundException("Student not found with id: " + dto.getId()));

        // Eski şifre doğrulaması
        if (dto.getOldPassword() != null && dto.getNewPassword() != null) {
            if (!passwordEncoder.matches(dto.getOldPassword(), existingStudent.getPassword())) {
                throw new BadRequestException("Eski şifre yanlış");
            }

            // Yeni şifreyi hashle
            String encodedPassword = passwordEncoder.encode(dto.getNewPassword());
            existingStudent.setPassword(encodedPassword);
            log.info("Password updated successfully");
        }

        // Diğer alanları güncelle
        if (dto.getUsername() != null) {
            existingStudent.setUsername(dto.getUsername());
        }
        if (dto.getEmail() != null) {
            existingStudent.setEmail(dto.getEmail());
        }
        if (dto.getFullName() != null) {
            existingStudent.setFullName(dto.getFullName());
        }
        if (dto.getPhone() != null) {
            existingStudent.setPhone(dto.getPhone());
        }

        Student savedStudent = studentRepository.save(existingStudent);

        // DTO'ya dönüştürürken şifreleri dahil etme
        StudentUpdateDTO result = new StudentUpdateDTO();
        result.setId(savedStudent.getId());
        result.setUsername(savedStudent.getUsername());
        result.setEmail(savedStudent.getEmail());
        result.setFullName(savedStudent.getFullName());
        result.setPhone(savedStudent.getPhone());

        return result;
    }
}