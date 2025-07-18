package com.smart.smartLibraryWeb.dto.studentDTO;


import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class StudentRegisterDTO {

    private String username;
    private String password;
    private String email;
    private String fullName;
    private String phone;

}
