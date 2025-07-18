package com.smart.smartLibraryWeb.controller;

import com.smart.smartLibraryWeb.dto.AuthRequest;
import com.smart.smartLibraryWeb.dto.AuthResponse;
import com.smart.smartLibraryWeb.model.Student;
import com.smart.smartLibraryWeb.service.AuthService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
@Slf4j
public class AuthController {

    private final AuthService authService;

    @PostMapping("/student/login")
    public ResponseEntity<?> studentLogin(@RequestBody AuthRequest authRequest) {
        try {
            log.info("Student login attempt for username: {}", authRequest.getUsername());

            // Giriş verilerini doğrula
            if (authRequest.getUsername() == null || authRequest.getUsername().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Username is required"));
            }

            if (authRequest.getPassword() == null || authRequest.getPassword().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Password is required"));
            }

            AuthResponse response = authService.studentLogin(authRequest);
            log.info("Student login successful for username: {}", authRequest.getUsername());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Student login failed for username: {}, error: {}", authRequest.getUsername(), e.getMessage());
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ErrorResponse("Login failed: " + e.getMessage()));
        }
    }

    @PostMapping("/admin/login")
    public ResponseEntity<?> adminLogin(@RequestBody AuthRequest authRequest) {
        try {
            log.info("Admin login attempt for username: {}", authRequest.getUsername());

            // Giriş verilerini doğrula
            if (authRequest.getUsername() == null || authRequest.getUsername().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Username is required"));
            }

            if (authRequest.getPassword() == null || authRequest.getPassword().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Password is required"));
            }

            AuthResponse response = authService.adminLogin(authRequest);
            log.info("Admin login successful for username: {}", authRequest.getUsername());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Admin login failed for username: {}, error: {}", authRequest.getUsername(), e.getMessage());
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(new ErrorResponse("Login failed: " + e.getMessage()));
        }
    }

    @PostMapping("/student/register")
    public ResponseEntity<?> registerStudent(@RequestBody Student student) {
        try {
            log.info("Student registration attempt for username: {}", student.getUsername());

            // Kayıt verilerini doğrula
            if (student.getUsername() == null || student.getUsername().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Username is required"));
            }

            if (student.getPassword() == null || student.getPassword().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Password is required"));
            }

            if (student.getEmail() == null || student.getEmail().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Email is required"));
            }

            if (student.getFullName() == null || student.getFullName().trim().isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Full name is required"));
            }

            AuthResponse response = authService.registerStudent(student);
            log.info("Student registration successful for username: {}", student.getUsername());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("Student registration failed for username: {}, error: {}", student.getUsername(), e.getMessage());
            return ResponseEntity.badRequest()
                    .body(new ErrorResponse("Registration failed: " + e.getMessage()));
        }
    }

    // Hata mesajları için basit bir sınıf
    public static class ErrorResponse {
        private String message;

        public ErrorResponse(String message) {
            this.message = message;
        }

        public String getMessage() {
            return message;
        }

        public void setMessage(String message) {
            this.message = message;
        }
    }
}