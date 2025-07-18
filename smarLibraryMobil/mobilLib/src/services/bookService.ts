// src/services/BookService.ts
import axios from '../utils/axiosConfig';
import {
    BookFavoriteDTO,
    BookCreateDTO,
    BookUpdateDTO,
    BookListDTO,
    BookDetailDTO,
    BookRatingRequestDTO,
    BookRatingResponseDTO,
    BookRatingDetailDTO
} from '../types/book';

const BASE_URL = '/api/books';

// Kitap CRUD işlemleri
export const createBook = async (data: BookCreateDTO): Promise<void> => {
    await axios.post(BASE_URL, data);
};

export const updateBook = async (id: number, data: BookUpdateDTO): Promise<void> => {
    await axios.put(`${BASE_URL}/${id}`, data);
};

export const deleteBook = async (id: number): Promise<void> => {
    await axios.delete(`${BASE_URL}/${id}`);
};

// Kitap getirme işlemleri
export const getAllBooks = async (): Promise<BookListDTO[]> => {
    const res = await axios.get<BookListDTO[]>(BASE_URL);
    return res.data;
};

// 🔹 Backend'de /id/{id} endpoint'i kullanıyor
export const getBookById = async (id: number): Promise<BookDetailDTO> => {
    const res = await axios.get<BookDetailDTO>(`${BASE_URL}/id/${id}`);
    return res.data;
};

export const getBooksByCategory = async (category: string): Promise<BookListDTO[]> => {
    const res = await axios.get<BookListDTO[]>(`${BASE_URL}/category/${category}`);
    return res.data;
};

export const searchBooksByTitle = async (title: string): Promise<BookListDTO[]> => {
    const res = await axios.get<BookListDTO[]>(`${BASE_URL}/search`, {
        params: { title }
    });
    return res.data;
};

// ⭐ Rating işlemleri
export const rateBook = async (data: BookRatingRequestDTO): Promise<BookRatingResponseDTO> => {
    const res = await axios.post<BookRatingResponseDTO>(`${BASE_URL}/rate`, data);
    return res.data;
};

export const getAverageRating = async (bookId: number): Promise<number> => {
    const res = await axios.get<number>(`${BASE_URL}/${bookId}/average-rating`);
    return res.data;
};

// 🔹 Backend'de query parameter olarak userId bekliyor
export const getUserRating = async (bookId: number, userId: number): Promise<number | null> => {
    const res = await axios.get<number | null>(`${BASE_URL}/${bookId}/user-rating`, {
        params: { userId }
    });
    return res.data;
};

export const getTotalRatingsCount = async (bookId: number): Promise<number> => {
    const res = await axios.get<number>(`${BASE_URL}/${bookId}/total-ratings`);
    return res.data;
};

export const getBookRatingDetail = async (bookId: number): Promise<BookRatingDetailDTO> => {
    const res = await axios.get<BookRatingDetailDTO>(`${BASE_URL}/${bookId}/rating-detail`);
    return res.data;
};

export const getTotalRatings = async (bookId: number): Promise<number> => {
    try {
        const res = await axios.get<number>(`/api/books/${bookId}/total-ratings`);
        return res.data || 0;
    } catch (error) {
        console.error('Toplam puan sayısı alma hatası:', error);
        return 0;
    }
};

export const commentBook = async (data: BookRatingRequestDTO): Promise<BookRatingResponseDTO> => {
    const res = await axios.post<BookRatingResponseDTO>(`${BASE_URL}/comment`, data);
    return res.data;
};

export const getBookComments = async (bookId: number): Promise<BookRatingResponseDTO[]> => {
    const res = await axios.get<BookRatingResponseDTO[]>(`${BASE_URL}/${bookId}/comments`);
    return res.data;
};

// Favori kitapları listele
export const getFavoriteBooksByStudentId = async (studentId: number): Promise<BookFavoriteDTO[]> => {
    const res = await axios.get<BookFavoriteDTO[]>(`${BASE_URL}/list/${studentId}`);
    return res.data;
};

// Favori kitaba ekle
export const addFavoriteBook = async (studentId: number, bookId: number): Promise<BookFavoriteDTO> => {
    const res = await axios.post<BookFavoriteDTO>(`${BASE_URL}/add`, null, {
        params: { studentId, bookId }
    });
    return res.data;
};

// Favori kitaptan çıkar
export const removeFavoriteBook = async (studentId: number, bookId: number): Promise<BookFavoriteDTO> => {
    const res = await axios.delete<BookFavoriteDTO>(`${BASE_URL}/remove`, {
        params: { studentId, bookId }
    });
    return res.data;
};