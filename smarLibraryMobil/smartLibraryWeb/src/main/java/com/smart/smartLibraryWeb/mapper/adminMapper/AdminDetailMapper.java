package com.smart.smartLibraryWeb.mapper.adminMapper;


import com.smart.smartLibraryWeb.dto.adminDTO.AdminDetailDTO;
import com.smart.smartLibraryWeb.mapper.BaseMapper;
import com.smart.smartLibraryWeb.model.Admin;
import org.mapstruct.Mapper;
import org.mapstruct.ReportingPolicy;

@Mapper(unmappedTargetPolicy = ReportingPolicy.IGNORE,componentModel = "spring")
public interface AdminDetailMapper extends BaseMapper<Admin, AdminDetailDTO> {
    @Override
    AdminDetailDTO mapToDto(Admin entity);

    @Override
    Admin mapToEntity(AdminDetailDTO dto);
}
