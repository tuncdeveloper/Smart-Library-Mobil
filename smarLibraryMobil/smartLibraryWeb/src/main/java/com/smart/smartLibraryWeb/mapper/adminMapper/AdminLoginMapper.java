package com.smart.smartLibraryWeb.mapper.adminMapper;


import com.smart.smartLibraryWeb.dto.adminDTO.AdminLoginDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Admin;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;
import org.springframework.context.annotation.Configuration;


@Configuration
@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface AdminLoginMapper extends BaseMapper<Admin,AdminLoginDTO> {

    @Override
    Admin mapToEntity(AdminLoginDTO dto);

    @Override
    AdminLoginDTO mapToDto(Admin entity);
}
