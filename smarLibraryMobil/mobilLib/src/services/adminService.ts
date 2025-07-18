import axios from '../utils/axiosConfig';
import { AdminLoginDTO, AdminDetailDTO } from '../types/admin';

// ✅ Auth işlemleri için ayrı bir base url tanımlandı
const AUTH_BASE_URL = '/api/auth';
const ADMIN_BASE_URL = '/api/admin';

// ✅ Admin login backend'e göre güncellendi
export const adminLogin = async (data: AdminLoginDTO): Promise<string> => {
    const response = await axios.post<string>(`${AUTH_BASE_URL}/admin/login`, data);
    return response.data; // "Giriş başarılı" veya hata mesajı
};

// ✅ Admin profil bilgisi için (örneğin dashboard sayfası)
export const getAdminProfile = async (username: string): Promise<AdminDetailDTO> => {
    const response = await axios.get<AdminDetailDTO>(`${ADMIN_BASE_URL}/profile/${username}`);
    return response.data;
};
