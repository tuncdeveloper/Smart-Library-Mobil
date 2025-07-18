// Admin giriş isteği
export type AdminLoginDTO = {
    username: string;
    password: string;
};

// Admin giriş cevabı - Backend'deki AuthResponse'a göre
export type AdminLoginResponseDTO = {
    token: string;
    id: number;
    username: string;
    fullName: string;
    email: string;
    phone: string;
    userType: string;
};

// Admin profil detayları (Backend'deki AdminDetailDTO)
export type AdminDetailDTO = {
    id: number;
    fullName: string;
    email: string;
    phone: string;
    position: string;
    createdAt: string; // ISO string format
    isActive: boolean;
};
