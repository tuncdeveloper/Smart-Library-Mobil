package com.smart.smartLibraryWeb.service.adminSevice;


import com.smart.smartLibraryWeb.dto.adminDTO.AdminDetailDTO;
import com.smart.smartLibraryWeb.dto.adminDTO.AdminLoginDTO;

public interface AdminService {
    boolean login(AdminLoginDTO loginDTO);
    AdminDetailDTO getProfileByUsername(String username);
}

