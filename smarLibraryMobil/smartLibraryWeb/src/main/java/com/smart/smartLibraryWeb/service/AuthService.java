package com.smart.smartLibraryWeb.service;

import com.smart.smartLibraryWeb.dto.AuthRequest;
import com.smart.smartLibraryWeb.dto.AuthResponse;
import com.smart.smartLibraryWeb.model.Admin;
import com.smart.smartLibraryWeb.model.Student;
import com.smart.smartLibraryWeb.repository.AdminRepository;
import com.smart.smartLibraryWeb.repository.StudentRepository;
import com.smart.smartLibraryWeb.security.JwtUtil;
import lombok.RequiredArgsConstructor;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AuthService {

    private final StudentRepository studentRepository;
    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    public AuthResponse studentLogin(AuthRequest authRequest) {
        try {
            // Önce öğrenci varlığını kontrol et
            Student student = studentRepository.findByUsername(authRequest.getUsername())
                    .orElseThrow(() -> new RuntimeException("Student not found"));

            // Şifre doğrulaması
            if (!passwordEncoder.matches(authRequest.getPassword(), student.getPassword())) {
                throw new RuntimeException("Invalid password");
            }

            // JWT token oluştur
            String token = jwtUtil.generateToken(student.getUsername(), "STUDENT", student.getId());

            return new AuthResponse(token, student.getId(), student.getUsername(),
                    student.getFullName(), student.getEmail(),student.getPhone(), "STUDENT");
        } catch (Exception e) {
            throw new RuntimeException("Login failed: " + e.getMessage());
        }
    }

    public AuthResponse adminLogin(AuthRequest authRequest) {
        try {
            // Önce admin varlığını kontrol et
            Admin admin = adminRepository.findByUsername(authRequest.getUsername())
                    .orElseThrow(() -> new RuntimeException("Admin not found"));



            // Şifre doğrulaması
            if (!passwordEncoder.matches(authRequest.getPassword(), admin.getPassword())) {
                throw new RuntimeException("Invalid password");
            }

            // JWT token oluştur
            String token = jwtUtil.generateToken(admin.getUsername(), "ADMIN", admin.getId());

            return new AuthResponse(token, admin.getId(), admin.getUsername(),
                    admin.getFullName(), admin.getEmail(),admin.getPhone(), "ADMIN");
        } catch (Exception e) {
            throw new RuntimeException("Login failed: " + e.getMessage());
        }
    }

    public AuthResponse registerStudent(Student student) {
        if (studentRepository.existsByUsername(student.getUsername())) {
            throw new RuntimeException("Username already exists");
        }

        if (studentRepository.existsByEmail(student.getEmail())) {
            throw new RuntimeException("Email already exists");
        }

        student.setPassword(passwordEncoder.encode(student.getPassword()));
        Student savedStudent = studentRepository.save(student);

        String token = jwtUtil.generateToken(savedStudent.getUsername(), "STUDENT", savedStudent.getId());

        return new AuthResponse(token, savedStudent.getId(), savedStudent.getUsername(),
                savedStudent.getFullName(), savedStudent.getEmail(),student.getPhone(), "STUDENT");
    }
}