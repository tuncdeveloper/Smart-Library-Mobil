import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Alert,
    ActivityIndicator,
    ScrollView,
    KeyboardAvoidingView,
    Platform,
    Animated,
    Dimensions
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '@/src/contexts/AuthContext';
import { StudentLoginDTO } from '../types/student';
import { loginStudent } from '../services/studentService';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';

// Navigation tipini tanımlayın
type AuthStackParamList = {
    Login: undefined;
    Register: undefined;
};

type LoginScreenNavigationProp = NativeStackNavigationProp<AuthStackParamList, 'Login'>;

interface Props {
    onGoToRegister?: () => void; // Eski prop'u optional yapıyoruz
}

export default function LoginPage({ onGoToRegister }: Props) {
    const navigation = useNavigation<LoginScreenNavigationProp>();
    const { login } = useAuth();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [focusedInput, setFocusedInput] = useState<string | null>(null);

    const handleLogin = async () => {
        if (!username.trim() || !password.trim()) {
            Alert.alert('Hata', 'Lütfen kullanıcı adı ve şifre giriniz.');
            return;
        }

        const loginData: StudentLoginDTO = {
            username: username.trim(),
            password,
        };

        setIsLoading(true);
        try {
            const response = await loginStudent(loginData);
            login(response);
        } catch (error: any) {
            const errorMessage =
                error.response?.data?.message || error.message || 'Giriş işlemi başarısız oldu';
            Alert.alert('Giriş Hatası', errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    const handleGoToRegister = () => {
        if (onGoToRegister) {
            onGoToRegister();
        } else {
            navigation.navigate('Register');
        }
    };

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

            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={styles.keyboardContainer}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContainer}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                >
                    <View style={styles.formContainer}>
                        {/* Header Section */}
                        <View style={styles.header}>
                            <View style={styles.iconContainer}>
                                <Text style={styles.icon}>🚀</Text>
                            </View>
                            <Text style={styles.title}>Hoş Geldiniz</Text>
                            <Text style={styles.subtitle}>
                                Hesabınıza giriş yapın ve öğrenci portalınıza erişin
                            </Text>
                        </View>

                        {/* Input Section */}
                        <View style={styles.inputSection}>
                            <View style={styles.inputWrapper}>
                                <View style={styles.inputIconContainer}>
                                    <Text style={styles.inputIcon}>👤</Text>
                                </View>
                                <TextInput
                                    style={getInputStyle('username')}
                                    placeholder="Kullanıcı Adı"
                                    value={username}
                                    onChangeText={setUsername}
                                    autoCapitalize="none"
                                    placeholderTextColor="#a0aec0"
                                    onFocus={() => setFocusedInput('username')}
                                    onBlur={() => setFocusedInput(null)}
                                />
                            </View>

                            <View style={styles.inputWrapper}>
                                <View style={styles.inputIconContainer}>
                                    <Text style={styles.inputIcon}>🔒</Text>
                                </View>
                                <TextInput
                                    style={getInputStyle('password')}
                                    placeholder="Şifre"
                                    value={password}
                                    onChangeText={setPassword}
                                    secureTextEntry
                                    placeholderTextColor="#a0aec0"
                                    onFocus={() => setFocusedInput('password')}
                                    onBlur={() => setFocusedInput(null)}
                                />
                            </View>
                        </View>

                        {/* Button Section */}
                        <View style={styles.buttonSection}>
                            <TouchableOpacity
                                style={[styles.primaryButton, isLoading && styles.disabledButton]}
                                onPress={handleLogin}
                                disabled={isLoading}
                                activeOpacity={0.8}
                            >
                                <LinearGradient
                                    colors={isLoading ? ['#95a5a6', '#95a5a6'] : ['#667eea', '#764ba2']}
                                    style={styles.buttonGradient}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 0 }}
                                >
                                    {isLoading ? (
                                        <ActivityIndicator color="#fff" size="small" />
                                    ) : (
                                        <Text style={styles.primaryButtonText}>Giriş Yap</Text>
                                    )}
                                </LinearGradient>
                            </TouchableOpacity>

                            <View style={styles.dividerContainer}>
                                <View style={styles.dividerLine} />
                                <Text style={styles.dividerText}>veya</Text>
                                <View style={styles.dividerLine} />
                            </View>

                            <TouchableOpacity
                                style={styles.secondaryButton}
                                onPress={handleGoToRegister}
                                activeOpacity={0.8}
                            >
                                <Text style={styles.secondaryButtonText}>✨ Yeni Hesap Oluştur</Text>
                            </TouchableOpacity>
                        </View>

                        {/* Footer */}
                        <View style={styles.footer}>
                            <Text style={styles.helpText}>
                                Giriş yapmakta sorun mu yaşıyorsunuz?{' '}
                                <Text style={styles.link}>Yardım Alın</Text>
                            </Text>
                        </View>
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
        minHeight: height,
    },
    formContainer: {
        width: '95%',
        maxWidth: 400,
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
        width: 200,
        height: 200,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderRadius: 100,
        top: -100,
        right: -100,
        opacity: 0.6,
    },
    floatingElement2: {
        position: 'absolute',
        width: 150,
        height: 150,
        backgroundColor: 'rgba(255, 255, 255, 0.08)',
        borderRadius: 75,
        bottom: -75,
        left: -75,
        opacity: 0.4,
    },
    floatingElement3: {
        position: 'absolute',
        width: 100,
        height: 100,
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
        borderRadius: 50,
        top: '40%',
        left: -50,
        opacity: 0.3,
    },
    header: {
        alignItems: 'center',
        marginBottom: 35,
    },
    iconContainer: {
        width: 80,
        height: 80,
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        borderRadius: 40,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
        shadowColor: '#667eea',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
        elevation: 10,
    },
    icon: {
        fontSize: 36,
    },
    title: {
        fontSize: 32,
        fontWeight: '800',
        textAlign: 'center',
        marginBottom: 8,
        color: '#ffffff',
        textShadowColor: 'rgba(0, 0, 0, 0.3)',
        textShadowOffset: { width: 0, height: 2 },
        textShadowRadius: 4,
    },
    subtitle: {
        fontSize: 16,
        textAlign: 'center',
        color: 'rgba(255, 255, 255, 0.9)',
        lineHeight: 24,
        marginHorizontal: 10,
    },
    inputSection: {
        marginBottom: 30,
    },
    inputWrapper: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        borderRadius: 15,
        marginBottom: 20,
        paddingHorizontal: 15,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.1,
        shadowRadius: 15,
        elevation: 5,
    },
    inputIconContainer: {
        width: 40,
        height: 40,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: 10,
    },
    inputIcon: {
        fontSize: 20,
        opacity: 0.6,
    },
    input: {
        flex: 1,
        paddingVertical: 18,
        fontSize: 16,
        color: '#2c3e50',
        fontWeight: '500',
    },
    inputFocused: {
        color: '#667eea',
    },
    buttonSection: {
        marginBottom: 25,
    },
    primaryButton: {
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
    primaryButtonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '700',
        letterSpacing: 0.5,
    },
    secondaryButton: {
        borderWidth: 2,
        borderColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: 15,
        paddingVertical: 18,
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        shadowColor: '#fff',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.2,
        shadowRadius: 10,
        elevation: 5,
    },
    secondaryButtonText: {
        color: '#ffffff',
        fontSize: 17,
        fontWeight: '600',
        letterSpacing: 0.3,
    },
    dividerContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: 25,
    },
    dividerLine: {
        flex: 1,
        height: 1,
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
    dividerText: {
        marginHorizontal: 15,
        color: 'rgba(255, 255, 255, 0.8)',
        fontSize: 14,
        fontWeight: '500',
    },
    footer: {
        alignItems: 'center',
    },
    helpText: {
        textAlign: 'center',
        fontSize: 14,
        color: 'rgba(255, 255, 255, 0.8)',
        lineHeight: 22,
    },
    link: {
        color: '#ffffff',
        fontWeight: '700',
        textDecorationLine: 'underline',
    },
});