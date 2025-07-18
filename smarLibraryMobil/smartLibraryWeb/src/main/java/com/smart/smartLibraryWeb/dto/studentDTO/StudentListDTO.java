package com.smart.smartLibraryWeb.dto.studentDTO;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class StudentListDTO {

    private Long id;
    private String fullName;
    private String email;

}
