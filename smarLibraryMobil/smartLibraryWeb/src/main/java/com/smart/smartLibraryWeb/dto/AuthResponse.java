package com.smart.smartLibraryWeb.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor

public class AuthResponse {
   private String token;
   private Long id;
   private String username;
   private String fullName;
   private String email;
   private String phone;
   private String userType; // Ekle



   // Getter'lar ve setter'lar
}

