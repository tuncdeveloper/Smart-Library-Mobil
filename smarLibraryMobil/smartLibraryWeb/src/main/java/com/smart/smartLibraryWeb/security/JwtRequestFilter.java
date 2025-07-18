package com.smart.smartLibraryWeb.security;

import com.smart.smartLibraryWeb.service.CustomUserDetailsService;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class JwtRequestFilter extends OncePerRequestFilter {

    private final CustomUserDetailsService userDetailsService;
    private final JwtUtil jwtUtil;

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {

        // Login endpointleri için JWT kontrolünü atla
        String requestURI = request.getRequestURI();
        if (requestURI.contains("/api/auth/")) {
            chain.doFilter(request, response);
            return;
        }

        final String requestTokenHeader = request.getHeader("Authorization");

        String username = null;
        String jwtToken = null;
        String userType = null;
        List<String> authorities = null;

        if (requestTokenHeader != null && requestTokenHeader.startsWith("Bearer ")) {
            jwtToken = requestTokenHeader.substring(7);
            try {
                username = jwtUtil.extractUsername(jwtToken);
                userType = jwtUtil.extractUserType(jwtToken);
                authorities = jwtUtil.extractAuthorities(jwtToken);
            } catch (Exception e) {
                logger.error("JWT Token extraction failed", e);
            }
        }

        // Kullanıcı adı ve tipi null değilse ve kimlik doğrulama yapılmamışsa
        if (username != null && userType != null && SecurityContextHolder.getContext().getAuthentication() == null) {
            try {
                // Token hem imza hem süre hem de kullanıcı adı için doğrulanıyor
                if (jwtUtil.validateToken(jwtToken, username)) {

                    // Authorities'yi JWT'den al, yoksa CustomUserDetailsService'den al
                    UserDetails userDetails;
                    if (authorities != null && !authorities.isEmpty()) {
                        // JWT'den authorities'yi kullan
                        List<SimpleGrantedAuthority> grantedAuthorities = authorities.stream()
                                .map(SimpleGrantedAuthority::new)
                                .collect(Collectors.toList());

                        userDetails = User.builder()
                                .username(username)
                                .password("") // JWT authentication'da password gerekli değil
                                .authorities(grantedAuthorities)
                                .build();
                    } else {
                        // Fallback: CustomUserDetailsService'yi kullan
                        userDetails = this.userDetailsService.loadUserByUsernameAndType(username, userType);
                    }

                    UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken =
                            new UsernamePasswordAuthenticationToken(
                                    userDetails,
                                    null,
                                    userDetails.getAuthorities()
                            );
                    usernamePasswordAuthenticationToken.setDetails(
                            new WebAuthenticationDetailsSource().buildDetails(request)
                    );
                    SecurityContextHolder.getContext().setAuthentication(usernamePasswordAuthenticationToken);
                }
            } catch (Exception e) {
                logger.error("Cannot set user authentication", e);
            }
        }
        chain.doFilter(request, response);
    }
}