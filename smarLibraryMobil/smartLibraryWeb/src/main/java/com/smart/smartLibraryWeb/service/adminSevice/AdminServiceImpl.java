package com.smart.smartLibraryWeb.service.adminSevice;



import com.smart.smartLibraryWeb.dto.adminDTO.AdminDetailDTO;
import com.smart.smartLibraryWeb.dto.adminDTO.AdminLoginDTO;
import com.smart.smartLibraryWeb.mapper.adminMapper.AdminDetailMapper;
import com.smart.smartLibraryWeb.mapper.adminMapper.AdminLoginMapper;
import com.smart.smartLibraryWeb.model.Admin;
import com.smart.smartLibraryWeb.repository.AdminRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class AdminServiceImpl implements AdminService {

    private final AdminRepository adminRepository;
    private final AdminLoginMapper adminLoginMapper;
    private final AdminDetailMapper adminDetailMapper;

    @Override
    public boolean login(AdminLoginDTO loginDTO) {
        return adminRepository.findByUsername(loginDTO.getUsername())
                .map(admin -> admin.getPassword().equals(loginDTO.getPassword()))
                .orElse(false);
    }

    @Override
    public AdminDetailDTO getProfileByUsername(String username) {
        Admin admin = adminRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("Admin bulunamadı"));
        return adminDetailMapper.mapToDto(admin);
    }
}

