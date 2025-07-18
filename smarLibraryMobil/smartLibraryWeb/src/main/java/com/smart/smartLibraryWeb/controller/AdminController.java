package com.smart.smartLibraryWeb.controller;


import com.smart.smartLibraryWeb.dto.adminDTO.AdminDetailDTO;
import com.smart.smartLibraryWeb.dto.adminDTO.AdminLoginDTO;
import com.smart.smartLibraryWeb.service.adminSevice.AdminService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
public class AdminController {

    private final AdminService adminService;


    // Kullanıcı adı ile admin profil bilgisi getirme (GET)
    @GetMapping("/profile/{username}")
    public ResponseEntity<AdminDetailDTO> getProfileByUsername(@PathVariable String username) {
        AdminDetailDTO adminDetail = adminService.getProfileByUsername(username);
        return ResponseEntity.ok(adminDetail);
    }
}
