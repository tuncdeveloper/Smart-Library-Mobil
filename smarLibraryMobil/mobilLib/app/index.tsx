import React from 'react';
import { AuthProvider } from '@/src/contexts/AuthContext';
import AppRoutes from '../src/routes/AppRoutes';
import Toast from 'react-native-toast-message';

export default function App() {
    return (
        <AuthProvider>
            <AppRoutes />
            <Toast />
        </AuthProvider>
    );
}
