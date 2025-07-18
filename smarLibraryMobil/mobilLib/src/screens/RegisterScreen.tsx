import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    StyleSheet,
    TouchableOpacity,
    ActivityIndicator,
    ScrollView,
    KeyboardAvoidingView,
    Platform,
    Dimensions
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { StudentRegisterDTO } from '../types/student';
import { registerStudent } from '../services/studentService';
import { useAuth } from '@/src/contexts/AuthContext';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

// Navigation tipini tanımlayın
type AuthStackParamList = {
    Login: undefined;
    Register: undefined;
};

type RegisterScreenNavigationProp = NativeStackNavigationProp<AuthStackParamList, 'Register'>;

interface Props {
    onGoToLogin?: () => void; // Eski prop'u optional yapıyoruz
}

export default function RegisterPage({ onGoToLogin }: Props) {
    const navigation = useNavigation<RegisterScreenNavigationProp>();
    const { login } = useAuth();
    const [form, setForm] = useState<StudentRegisterDTO>({
        username: '',
        password: '',
        fullName: '',
        email: '',
        phone: '',
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [focusedInput, setFocusedInput] = useState<string | null>(null);

    const handleChange = (field: keyof StudentRegisterDTO, value: string) => {
        setForm((prev) => ({ ...prev, [field]: value }));
    };

    const handleRegister = async () => {
        const { username, password, fullName, email, phone } = form;
        setError('');
        setLoading(true);

        if (!username || !password || !fullName || !email || !phone) {
            setError('Lütfen tüm alanları doldurun');
            setLoading(false);
            return;
        }

        try {
            const registeredStudent = await registerStudent(form);
            // Başarılı kayıt sonrası otomatik giriş yap
            login(registeredStudent);
            // Eğer otomatik giriş yapılmasını istemiyorsanız, aşağıdaki satırları kullanabilirsiniz:
            // handleGoToLogin(); // Giriş sayfasına yönlendir
        } catch (error: any) {
            console.error('Kayıt hatası:', error);
            const errorMessage = error.response?.data?.message || error.message || 'Bir hata oluştu.';
            setError(`Kayıt başarısız: ${errorMessage}`);
        } finally {
            setLoading(false);
        }
    };

    const handleGoToLogin = () => {
        if (onGoToLogin) {
            onGoToLogin();
        } else {
            navigation.navigate('Login');
        }
    };

    const formFields = [
        { name: 'username', label: 'Kullanıcı Adı', placeholder: 'Kullanıcı adınızı giriniz', icon: '👤' },
        { name: 'password', label: 'Parola', placeholder: 'Güçlü bir parola oluşturun', icon: '🔒', secure: true },
        { name: 'email', label: 'Email', placeholder: 'ornek@email.com', icon: '📧', keyboard: 'email-address' },
        { name: 'fullName', label: 'Ad Soyad', placeholder: 'Adınız ve soyadınız', icon: '🏷️' },
        { name: 'phone', label: 'Telefon', placeholder: '+90 5XX XXX XX XX', icon: '📱', keyboard: 'phone-pad' },
    ];

    const getInputStyle = (inputName: string) => [
        styles.input,
        focusedInput === inputName && styles.inputFocused
    ];

    return (
        <LinearGradient
            colors={['#667eea', '#764ba2', '#f093fb']}
            style={styles.container}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
        >
            {/* Floating Background Elements */}
            <View style={styles.floatingElement1} />
            <View style={styles.floatingElement2} />
            <View style={styles.floatingElement3} />
            <View style={styles.floatingElement4} />

            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={styles.keyboardContainer}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContainer}
                    showsVerticalScrollIndicator={false}
                >
                    <View style={styles.formContainer}>
                        {/* Header Section */}
                        <View style={styles.header}>
                            <View style={styles.iconContainer}>
                                <Text style={styles.icon}>✨</Text>
                            </View>
                            <Text style={styles.title}>Kayıt Ol</Text>
                            <Text style={styles.subtitle}>
                                Yeni hesap oluştur ve öğrenci portalına katıl
                            </Text>
                        </View>

                        {/* Error Message */}
                        {error ? (
                            <View style={styles.errorContainer}>
                                <Text style={styles.errorIcon}>⚠️</Text>
                                <Text style={styles.errorText}>{error}</Text>
                            </View>
                        ) : null}

                        {/* Input Fields */}
                        <View style={styles.inputSection}>
                            {formFields.map((field) => (
                                <View key={field.name} style={styles.inputWrapper}>
                                    <View style={styles.inputIconContainer}>
                                        <Text style={styles.inputIcon}>{field.icon}</Text>
                                    </View>
                                    <View style={styles.inputContent}>
                                        <Text style={styles.inputLabel}>{field.label}</Text>
                                        <TextInput
                                            style={getInputStyle(field.name)}
                                            placeholder={field.placeholder}
                                            placeholderTextColor="#a0aec0"
                                            value={form[field.name as keyof StudentRegisterDTO]}
                                            onChangeText={(text) => handleChange(field.name as keyof StudentRegisterDTO, text)}
                                            secureTextEntry={field.secure}
                                            keyboardType={field.keyboard as any}
                                            autoCapitalize="none"
                                            onFocus={() => {
                                                setFocusedInput(field.name);
                                                setError('');
                                            }}
                                            onBlur={() => setFocusedInput(null)}
                                        />
                                    </View>
                                </View>
                            ))}
                        </View>

                        {/* Register Button */}
                        <TouchableOpacity
                            style={[styles.registerButton, loading && styles.disabledButton]}
                            onPress={handleRegister}
                            disabled={loading}
                            activeOpacity={0.8}
                        >
                            <LinearGradient
                                colors={loading ? ['#95a5a6', '#95a5a6'] : ['#667eea', '#764ba2']}
                                style={styles.buttonGradient}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 0 }}
                            >
                                {loading ? (
                                    <View style={styles.loadingContainer}>
                                        <ActivityIndicator size="small" color="#fff" />
                                        <Text style={styles.loadingText}>Kayıt yapılıyor...</Text>
                                    </View>
                                ) : (
                                    <View style={styles.buttonContent}>
                                        <Text style={styles.buttonText}>🚀 Kayıt Ol</Text>
                                    </View>
                                )}
                            </LinearGradient>
                        </TouchableOpacity>

                        {/* Login Link */}
                        <TouchableOpacity onPress={handleGoToLogin} style={styles.loginLink}>
                            <Text style={styles.loginText}>
                                Zaten hesabınız var mı?{' '}
                                <Text style={styles.loginLinkText}>Giriş Yapın</Text>
                            </Text>
                        </TouchableOpacity>
                    </View>
                </ScrollView>
            </KeyboardAvoidingView>
        </LinearGradient>
    );
}

const { width, height } = Dimensions.get('window');

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    keyboardContainer: {
        flex: 1,
    },
    scrollContainer: {
        flexGrow: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
        paddingTop: 60,
        paddingBottom: 40,
    },
    formContainer: {
        width: '95%',
        maxWidth: 420,
        backgroundColor: 'rgba(255, 255, 255, 0.25)',
        backdropFilter: 'blur(20px)',
        borderRadius: 25,
        padding: 30,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 20 },
        shadowOpacity: 0.15,
        shadowRadius: 40,
        elevation: 20,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    floatingElement1: {
        position: 'absolute',
        width: 180,
        height: 180,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: 90,
        top: -90,
        right: -90,
        opacity: 0.6,
    },
    floatingElement2: {
        position: 'absolute',
        width: 120,
        height: 120,
        backgroundColor: 'rgba(255, 255, 255, 0.08)',
        borderRadius: 60,
        bottom: -60,
        left: -60,
        opacity: 0.4,
    },
    floatingElement3: {
        position: 'absolute',
        width: 80,
        height: 80,
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        borderRadius: 40,
        top: '30%',
        left: -40,
        opacity: 0.3,
    },
    floatingElement4: {
        position: 'absolute',
        width: 60,
        height: 60,
        backgroundColor: 'rgba(255, 255, 255, 0.06)',
        borderRadius: 30,
        top: '60%',
        right: -30,
        opacity: 0.4,
    },
    header: {
        alignItems: 'center',
        marginBottom: 30,
    },
    iconContainer: {
        width: 70,
        height: 70,
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        borderRadius: 35,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 15,
        shadowColor: '#667eea',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 15,
        elevation: 8,
    },
    icon: {
        fontSize: 32,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        textAlign: 'center',
        marginBottom: 8,
        color: '#ffffff',
        textShadowColor: 'rgba(0, 0, 0, 0.3)',
        textShadowOffset: { width: 0, height: 2 },
        textShadowRadius: 4,
    },
    subtitle: {
        fontSize: 15,
        textAlign: 'center',
        color: 'rgba(255, 255, 255, 0.9)',
        lineHeight: 22,
        marginHorizontal: 10,
    },
    errorContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(231, 76, 60, 0.2)',
        borderRadius: 15,
        padding: 15,
        marginBottom: 25,
        borderWidth: 1,
        borderColor: 'rgba(231, 76, 60, 0.3)',
    },
    errorIcon: {
        fontSize: 20,
        marginRight: 10,
    },
    errorText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
        flex: 1,
    },
    inputSection: {
        marginBottom: 25,
    },
    inputWrapper: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        borderRadius: 15,
        marginBottom: 18,
        paddingHorizontal: 15,
        paddingVertical: 5,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 3,
    },
    inputIconContainer: {
        width: 35,
        height: 50,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 10,
    },
    inputIcon: {
        fontSize: 18,
        opacity: 0.6,
    },
    inputContent: {
        flex: 1,
    },
    inputLabel: {
        fontSize: 12,
        fontWeight: '600',
        color: '#667eea',
        marginBottom: 2,
        marginTop: 8,
    },
    input: {
        fontSize: 15,
        color: '#2c3e50',
        fontWeight: '500',
        paddingVertical: 8,
        paddingBottom: 12,
    },
    inputFocused: {
        color: '#667eea',
    },
    registerButton: {
        borderRadius: 15,
        overflow: 'hidden',
        marginBottom: 20,
        shadowColor: '#667eea',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.4,
        shadowRadius: 20,
        elevation: 10,
    },
    buttonGradient: {
        paddingVertical: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    disabledButton: {
        opacity: 0.7,
    },
    buttonContent: {
        alignItems: 'center',
    },
    buttonText: {
        color: '#fff',
        fontSize: 17,
        fontWeight: '700',
        letterSpacing: 0.5,
    },
    loadingContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    loadingText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
        marginLeft: 10,
    },
    loginLink: {
        alignItems: 'center',
        paddingVertical: 15,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: 15,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.2)',
    },
    loginText: {
        textAlign: 'center',
        color: 'rgba(255, 255, 255, 0.9)',
        fontSize: 14,
        fontWeight: '500',
    },
    loginLinkText: {
        color: '#ffffff',
        fontWeight: '700',
        textDecorationLine: 'underline',
    },
});