// src/contexts/AuthContext.tsx
import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { StudentLoginResponseDTO } from '../types/student';

interface AuthContextType {
    user: StudentLoginResponseDTO | null;
    login: (user: StudentLoginResponseDTO) => Promise<void>;
    logout: () => Promise<void>;
    isLoading: boolean;
    isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
    const [user, setUser] = useState<StudentLoginResponseDTO | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const loadUser = async () => {
            try {
                const storedUser = await AsyncStorage.getItem('student');
                if (storedUser) {
                    const parsedUser = JSON.parse(storedUser);

                    // Token var mı kontrol et
                    if (parsedUser.token) {
                        setUser(parsedUser);
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

    const login = async (userData: StudentLoginResponseDTO) => {
        setUser(userData);
        await AsyncStorage.setItem('student', JSON.stringify(userData));
    };

    const logout = async () => {
        setUser(null);
        await AsyncStorage.removeItem('student');
    };

    return (
        <AuthContext.Provider value={{
            user,
            login,
            logout,
            isLoading,
            isAuthenticated: !!user?.token
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