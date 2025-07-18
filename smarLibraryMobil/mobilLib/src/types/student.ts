// Öğrenci listesi için temel DTO
export type StudentListDTO = {
    id: number;
    fullName: string;
    email: string;
};

// Giriş yaparken gönderilen veri (AuthRequest.java karşılığı)
export type StudentLoginDTO = {
    username: string;
    password: string;
};

// Giriş sonrası dönen yanıt (AuthResponse.java karşılığı)
export type StudentLoginResponseDTO = {
    token: string;
    id: number;
    username: string;
    fullName: string;
    email: string;
    phone: string;
    userType: string;
};

// Öğrenci kayıt isteği
export type StudentRegisterDTO = {
    username: string;
    password: string;
    email: string;
    fullName: string;
    phone: string;
};

// 🔧 Öğrenci bilgilerini güncellemek için kullanılan DTO - password optional yapıldı
export type StudentUpdateDTO = {
    id: number;
    username: string;
    password?: string; // Optional - şifre güncellenirken kullanılır
    email: string;
    fullName: string;
    phone: string;
    oldPassword?: string; // Eski şifre için
    newPassword?: string; // Yeni şifre için
};