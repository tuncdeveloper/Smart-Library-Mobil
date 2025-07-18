package com.smart.smartLibraryWeb.service;

import com.smart.smartLibraryWeb.model.Admin;
import com.smart.smartLibraryWeb.model.Student;
import com.smart.smartLibraryWeb.repository.AdminRepository;
import com.smart.smartLibraryWeb.repository.StudentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.Collections;

@Service
@RequiredArgsConstructor
public class CustomUserDetailsService implements UserDetailsService {

    private final StudentRepository studentRepository;
    private final AdminRepository adminRepository;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        throw new UnsupportedOperationException("Use loadUserByUsernameAndType method instead");
    }

    public UserDetails loadUserByUsernameAndType(String username, String userType) throws UsernameNotFoundException {
        // Kullanıcı tipini büyük harfe çevirerek standartlaştır
        String normalizedType = userType.toUpperCase();

        if ("STUDENT".equals(normalizedType)) {
            Student student = studentRepository.findByUsername(username)
                    .orElseThrow(() -> new UsernameNotFoundException("Student not found: " + username));



            return new User(
                    student.getUsername(),
                    student.getPassword(),
                    Collections.singletonList(new SimpleGrantedAuthority("ROLE_STUDENT"))
            );
        } else if ("ADMIN".equals(normalizedType)) {
            Admin admin = adminRepository.findByUsername(username)
                    .orElseThrow(() -> new UsernameNotFoundException("Admin not found: " + username));



            return new User(
                    admin.getUsername(),
                    admin.getPassword(),
                    Collections.singletonList(new SimpleGrantedAuthority("ROLE_ADMIN"))
            );
        }

        throw new UsernameNotFoundException("Invalid user type: " + userType);
    }
}