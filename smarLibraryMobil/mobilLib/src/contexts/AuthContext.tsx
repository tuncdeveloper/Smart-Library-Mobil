// src/contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StudentLoginResponseDTO } from '../types/student';
import { BookFavoriteDTO } from '../types/book';
import { getFavoriteBooksByStudentId, addFavoriteBook, removeFavoriteBook } from '../services/bookService';

interface AuthContextType {
    user: StudentLoginResponseDTO | null;
    login: (user: StudentLoginResponseDTO) => Promise<void>;
    logout: () => Promise<void>;
    isLoading: boolean;
    isAuthenticated: boolean;
    // Favori yönetimi
    favorites: Set<number>;
    favoriteBooks: BookFavoriteDTO[];
    favoritesLoading: boolean;
    addToFavorites: (bookId: number) => Promise<boolean>;
    removeFromFavorites: (bookId: number) => Promise<boolean>;
    refreshFavorites: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<StudentLoginResponseDTO | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    // Favori yönetimi state'leri
    const [favorites, setFavorites] = useState<Set<number>>(new Set());
    const [favoriteBooks, setFavoriteBooks] = useState<BookFavoriteDTO[]>([]);
    const [favoritesLoading, setFavoritesLoading] = useState(false);

    // Kullanıcı yüklendiğinde favorileri de yükle
    useEffect(() => {
        const loadUser = async () => {
            try {
                const storedUser = await AsyncStorage.getItem('student');
                if (storedUser) {
                    const parsedUser = JSON.parse(storedUser);

                    // Token var mı kontrol et
                    if (parsedUser.token) {
                        setUser(parsedUser);
                        // Kullanıcı yüklendikten sonra favorileri yükle
                        await loadFavorites(parsedUser.id);
                    } else {
                        console.warn('Token bulunamadı, kullanıcı çıkış yapılıyor');
                        await AsyncStorage.removeItem('student');
                    }
                }
            } catch (error) {
                console.error('AsyncStorage\'dan kullanıcı yüklenemedi:', error);
                await AsyncStorage.removeItem('student');
            } finally {
                setIsLoading(false);
            }
        };

        loadUser();
    }, []);

    // Favorileri yükle
    const loadFavorites = async (userId: number) => {
        if (!userId) return;

        try {
            setFavoritesLoading(true);
            const favBooks: BookFavoriteDTO[] = await getFavoriteBooksByStudentId(userId);
            const favIds = new Set(favBooks.map(fav => fav.bookId));

            setFavorites(favIds);
            setFavoriteBooks(favBooks);
        } catch (error) {
            console.error('Favoriler yüklenirken hata:', error);
        } finally {
            setFavoritesLoading(false);
        }
    };

    const login = async (userData: StudentLoginResponseDTO) => {
        setUser(userData);
        await AsyncStorage.setItem('student', JSON.stringify(userData));
        // Login sonrası favorileri yükle
        await loadFavorites(userData.id);
    };

    const logout = async () => {
        setUser(null);
        setFavorites(new Set());
        setFavoriteBooks([]);
        await AsyncStorage.removeItem('student');
    };

    // Favorilere ekleme
    const addToFavorites = async (bookId: number): Promise<boolean> => {
        if (!user?.id) {
            return false;
        }

        try {
            await addFavoriteBook(user.id, bookId);

            // State'i güncelle
            setFavorites(prev => new Set([...prev, bookId]));

            // Favori kitaplar listesini yeniden yükle
            await refreshFavorites();

            return true;
        } catch (error) {
            console.error('Favori ekleme hatası:', error);
            return false;
        }
    };

    // Favorilerden çıkarma
    const removeFromFavorites = async (bookId: number): Promise<boolean> => {
        if (!user?.id) {
            return false;
        }

        try {
            await removeFavoriteBook(user.id, bookId);

            // State'i güncelle
            setFavorites(prev => new Set([...prev].filter(id => id !== bookId)));
            setFavoriteBooks(prev => prev.filter(book => book.bookId !== bookId));

            return true;
        } catch (error) {
            console.error('Favori çıkarma hatası:', error);
            return false;
        }
    };

    // Favorileri yenile
    const refreshFavorites = async () => {
        if (user?.id) {
            await loadFavorites(user.id);
        }
    };

    return (
        <AuthContext.Provider value={{
            user,
            login,
            logout,
            isLoading,
            isAuthenticated: !!user?.token,
            favorites,
            favoriteBooks,
            favoritesLoading,
            addToFavorites,
            removeFromFavorites,
            refreshFavorites
        }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};