import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const instance = axios.create({
    baseURL: 'http://192.168.1.8:8080',
    timeout: 5000,
});

// Request Interceptor (Token Ekleme)
instance.interceptors.request.use(
    async (config) => {
        const tokenString = await AsyncStorage.getItem('student');
        if (tokenString) {
            try {
                const user = JSON.parse(tokenString);
                if (user.token) {
                    config.headers = config.headers || {};
                    config.headers.Authorization = `Bearer ${user.token}`;
                }
            } catch (error) {
                console.error('Token parse hatası:', error);
                await AsyncStorage.removeItem('student');
            }
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response Interceptor (401 → logout)
instance.interceptors.response.use(
    (response) => response,
    async (error) => {
        if (error.response?.status === 401) {
            await AsyncStorage.removeItem('student');

            // Geriye yönlendirme burada yapılamaz: navigation burada erişilebilir değil.
            // Bunun yerine aşağıdaki yöntem ile ayrı bir logout yardımcı fonksiyonu çağrılabilir.
        }
        return Promise.reject(error);
    }
);

export default instance;
