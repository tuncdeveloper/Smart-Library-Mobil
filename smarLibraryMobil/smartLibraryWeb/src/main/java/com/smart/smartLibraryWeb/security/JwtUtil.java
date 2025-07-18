package com.smart.smartLibraryWeb.security;

import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

@Component
public class JwtUtil {

    @Value("${jwt.secret}")
    private String secret;

    @Value("${jwt.expiration}")
    private long expiration;

    private SecretKey getSigningKey() {
        byte[] keyBytes = secret.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }

    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }

    public String extractUserType(String token) {
        return extractClaim(token, claims -> claims.get("userType", String.class));
    }

    public Long extractUserId(String token) {
        return extractClaim(token, claims -> claims.get("userId", Long.class));
    }

    // Authorities'yi JWT'den çıkar
    public List<String> extractAuthorities(String token) {
        return extractClaim(token, claims -> claims.get("authorities", List.class));
    }

    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }

    private Claims extractAllClaims(String token) {
        return Jwts.parserBuilder()
                .setSigningKey(getSigningKey())
                .build()
                .parseClaimsJws(token)
                .getBody();
    }

    // Güncellenmiş token generation method - authorities ile
    public String generateToken(String username, String userType, Long userId) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userType", userType);
        claims.put("userId", userId);

        // UserType'a göre authority ekle
        List<String> authorities;
        if ("STUDENT".equals(userType.toUpperCase())) {
            authorities = List.of("ROLE_STUDENT");
        } else if ("ADMIN".equals(userType.toUpperCase())) {
            authorities = List.of("ROLE_ADMIN");
        } else {
            authorities = List.of("ROLE_USER");
        }

        claims.put("authorities", authorities);
        return createToken(claims, username);
    }

    // Overloaded method - authorities'yi direkt alır
    public String generateToken(String username, String userType, Long userId, List<String> authorities) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userType", userType);
        claims.put("userId", userId);
        claims.put("authorities", authorities);
        return createToken(claims, username);
    }

    private String createToken(Map<String, Object> claims, String subject) {
        return Jwts.builder()
                .setClaims(claims)
                .setSubject(subject)
                .setIssuedAt(new Date(System.currentTimeMillis()))
                .setExpiration(new Date(System.currentTimeMillis() + expiration))
                .signWith(getSigningKey(), SignatureAlgorithm.HS256)
                .compact();
    }

    public Boolean validateToken(String token, String username) {
        try {
            final Claims claims = Jwts.parserBuilder()
                    .setSigningKey(getSigningKey())
                    .build()
                    .parseClaimsJws(token)
                    .getBody();

            return claims.getSubject().equals(username) && !claims.getExpiration().before(new Date());
        } catch (Exception e) {
            return false;
        }
    }
}