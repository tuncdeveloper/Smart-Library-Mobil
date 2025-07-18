package com.smart.smartLibraryWeb.dto.studentDTO;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor

public class StudentUpdateDTO {

    private Long id;
    private String username;
    private String email;
    private String fullName;
    private String phone;
    private String oldPassword;   // Eski şifre için
    private String newPassword;   // Yeni şifre için

    // password güncelleme ayrı DTO veya endpoint ile yapılabilir.

}

