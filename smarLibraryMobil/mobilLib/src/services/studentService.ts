import axios from '../utils/axiosConfig';
import {
    StudentListDTO,
    StudentLoginDTO,
    StudentRegisterDTO,
    StudentLoginResponseDTO,
    StudentUpdateDTO
} from '../types/student';

const AUTH_BASE_URL = '/api/auth';
const STUDENT_BASE_URL = '/api/students';

export const registerStudent = async (data: StudentRegisterDTO): Promise<StudentLoginResponseDTO> => {
    const res = await axios.post<StudentLoginResponseDTO>(`${AUTH_BASE_URL}/student/register`, data);
    return res.data;
};

export const loginStudent = async (data: StudentLoginDTO): Promise<StudentLoginResponseDTO> => {
    const res = await axios.post<StudentLoginResponseDTO>(`${AUTH_BASE_URL}/student/login`, data);
    return res.data;
};

export const getAllStudents = async (): Promise<StudentListDTO[]> => {
    const res = await axios.get<StudentListDTO[]>(STUDENT_BASE_URL);
    return res.data;
};

export const getStudentById = async (id: number): Promise<StudentListDTO> => {
    const res = await axios.get<StudentListDTO>(`${STUDENT_BASE_URL}/${id}`);
    return res.data;
};

// 🔧 Güncelleme: Şifre alanı kontrolü ve hata yönetimi eklendi
export const updateStudent = async (id: number, data: StudentUpdateDTO): Promise<StudentUpdateDTO> => {
    try {
        console.log('Update request payload:', data);

        const res = await axios.put<StudentUpdateDTO>(`${STUDENT_BASE_URL}/${id}`, data);
        return res.data;
    } catch (error: any) {
        console.error('Update student error:', error);

        // Backend'den gelen özel hata mesajını kullan
        const errorMessage = error.response?.data?.message || error.message;

        if (error.response?.status === 404) {
            throw new Error('Öğrenci bulunamadı');
        } else if (error.response?.status === 400) {
            throw new Error(errorMessage || 'Geçersiz veri gönderildi');
        } else if (error.response?.status === 401) {
            throw new Error(errorMessage || 'Eski şifre yanlış');
        } else if (error.response?.status === 500) {
            throw new Error('Sunucu hatası oluştu');
        } else {
            throw new Error(`Güncelleme hatası: ${errorMessage}`);
        }
    }
};

// verifyPassword fonksiyonunu kaldırıyoruz, artık gerek yok

export const deleteStudent = async (id: number): Promise<string> => {
    const res = await axios.delete<string>(`${STUDENT_BASE_URL}/${id}`);
    return res.data;
};

