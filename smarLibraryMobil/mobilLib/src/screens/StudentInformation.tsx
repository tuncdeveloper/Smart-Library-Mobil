import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Alert,
    ScrollView,
    ActivityIndicator,
    Switch,
    KeyboardAvoidingView,
    Platform,
    Animated,
    Dimensions,
    StatusBar
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { updateStudent } from '../services/studentService';
import { StudentUpdateDTO } from '../types/student';

const { width, height } = Dimensions.get('window');

const StudentInformation: React.FC = () => {
    const { user, login } = useAuth();
    const [loading, setLoading] = useState(false);
    const [userData, setUserData] = useState({
        id: 0,
        username: '',
        fullName: '',
        email: '',
        phone: '',
        oldPassword: '',
        newPassword: '',
        confirmPassword: ''
    });

    const [errors, setErrors] = useState<{ [key: string]: string }>({});
    const [isPasswordUpdate, setIsPasswordUpdate] = useState(false);
    const [fadeAnim] = useState(new Animated.Value(0));
    const [slideAnim] = useState(new Animated.Value(50));

    useEffect(() => {
        if (user) {
            setUserData((prev) => ({
                ...prev,
                id: user.id,
                username: user.username,
                fullName: user.fullName,
                email: user.email,
                phone: user.phone
            }));
        }

        // Giriş animasyonu
        Animated.parallel([
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 800,
                useNativeDriver: true,
            }),
            Animated.timing(slideAnim, {
                toValue: 0,
                duration: 800,
                useNativeDriver: true,
            })
        ]).start();
    }, [user]);

    const handleInputChange = (field: string, value: string) => {
        setUserData((prev) => ({ ...prev, [field]: value }));
        if (errors[field]) {
            setErrors((prev) => ({ ...prev, [field]: '' }));
        }
    };

    const validateForm = (): boolean => {
        const newErrors: { [key: string]: string } = {};

        if (!userData.username.trim()) {
            newErrors.username = 'Kullanıcı adı gerekli';
        }

        if (!userData.fullName.trim()) {
            newErrors.fullName = 'Ad Soyad gerekli';
        }

        if (!userData.email.trim()) {
            newErrors.email = 'E-posta gerekli';
        } else if (!/\S+@\S+\.\S+/.test(userData.email)) {
            newErrors.email = 'Geçerli bir e-posta girin';
        }

        if (!userData.phone.trim()) {
            newErrors.phone = 'Telefon numarası gerekli';
        }

        if (isPasswordUpdate) {
            if (!userData.oldPassword.trim()) {
                newErrors.oldPassword = 'Eski şifre gerekli';
            }
            if (!userData.newPassword.trim()) {
                newErrors.newPassword = 'Yeni şifre gerekli';
            } else if (userData.newPassword.length < 6) {
                newErrors.newPassword = 'Yeni şifre en az 6 karakter olmalı';
            }
            if (userData.newPassword !== userData.confirmPassword) {
                newErrors.confirmPassword = 'Şifreler eşleşmiyor';
            }
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleUpdate = async () => {
        if (!validateForm()) return;

        setLoading(true);

        try {
            const updateData: StudentUpdateDTO = {
                id: userData.id,
                username: userData.username,
                fullName: userData.fullName,
                email: userData.email,
                phone: userData.phone
            };

            if (isPasswordUpdate) {
                updateData.oldPassword = userData.oldPassword;
                updateData.newPassword = userData.newPassword;
            }

            const updatedUser = await updateStudent(userData.id, updateData);

            // AuthContext'i güncelle
            if (user) {
                const updatedUserData = {
                    ...user,
                    username: updatedUser.username,
                    fullName: updatedUser.fullName,
                    email: updatedUser.email,
                    phone: updatedUser.phone
                };
                await login(updatedUserData);
            }

            Alert.alert('Başarılı', 'Bilgiler güncellendi');

            setUserData((prev) => ({
                ...prev,
                oldPassword: '',
                newPassword: '',
                confirmPassword: ''
            }));
            setIsPasswordUpdate(false);
        } catch (error: any) {
            console.error('Güncelleme hatası:', error);
            Alert.alert('Hata', error?.message || 'Güncelleme sırasında hata oluştu');
        } finally {
            setLoading(false);
        }
    };

    const renderInput = (
        label: string,
        field: string,
        placeholder: string,
        secureTextEntry = false,
        icon?: string
    ) => (
        <View style={styles.inputContainer}>
            <Text style={styles.label}>{label}</Text>
            <View style={styles.inputWrapper}>
                {icon && <Text style={styles.inputIcon}>{icon}</Text>}
                <TextInput
                    style={[
                        styles.input,
                        errors[field] && styles.inputError,
                        icon && styles.inputWithIcon
                    ]}
                    placeholder={placeholder}
                    value={userData[field as keyof typeof userData] as string}
                    onChangeText={(val) => handleInputChange(field, val)}
                    secureTextEntry={secureTextEntry}
                    placeholderTextColor="#94a3b8"
                />
            </View>
            {errors[field] && (
                <View style={styles.errorContainer}>
                    <Text style={styles.errorIcon}>⚠️</Text>
                    <Text style={styles.errorText}>{errors[field]}</Text>
                </View>
            )}
        </View>
    );

    // Kullanıcı yüklenmesini bekle
    if (!user) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2', '#f093fb']}
                style={styles.container}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            >
                <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />
                <View style={styles.loadingContainer}>
                    <View style={styles.loadingCard}>
                        <ActivityIndicator size="large" color="#667eea" />
                        <Text style={styles.loadingText}>Kullanıcı bilgileri yükleniyor...</Text>
                        <View style={styles.loadingDots}>
                            <View style={[styles.dot, styles.dot1]} />
                            <View style={[styles.dot, styles.dot2]} />
                            <View style={[styles.dot, styles.dot3]} />
                        </View>
                    </View>
                </View>
            </LinearGradient>
        );
    }

    return (
        <LinearGradient
            colors={['#667eea', '#764ba2', '#f093fb']}
            style={styles.container}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
        >
            <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />

            {/* Decorative Elements */}
            <View style={styles.decorativeCircle1} />
            <View style={styles.decorativeCircle2} />
            <View style={styles.decorativeCircle3} />

            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={styles.keyboardContainer}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContainer}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                >
                    <Animated.View
                        style={[
                            styles.formContainer,
                            {
                                opacity: fadeAnim,
                                transform: [{ translateY: slideAnim }]
                            }
                        ]}
                    >
                        <View style={styles.header}>
                            <View style={styles.headerIcon}>
                                <Text style={styles.headerIconText}>👤</Text>
                            </View>
                            <Text style={styles.headerTitle}>Profil Bilgileri</Text>
                            <Text style={styles.headerSubtitle}>Kişisel bilgilerinizi güncelleyin</Text>
                        </View>

                        <View style={styles.form}>
                            {renderInput('Kullanıcı Adı', 'username', 'Kullanıcı adınızı girin', false, '👤')}
                            {renderInput('Ad Soyad', 'fullName', 'Ad soyadınızı girin', false, '📝')}
                            {renderInput('E-posta', 'email', 'E-posta adresinizi girin', false, '📧')}
                            {renderInput('Telefon', 'phone', 'Telefon numaranızı girin', false, '📱')}

                            <View style={styles.switchContainer}>
                                <View style={styles.switchLabelContainer}>
                                    <Text style={styles.switchIcon}>🔐</Text>
                                    <Text style={styles.switchLabel}>Şifre Güncelle</Text>
                                </View>
                                <Switch
                                    value={isPasswordUpdate}
                                    onValueChange={setIsPasswordUpdate}
                                    trackColor={{ false: '#e2e8f0', true: '#667eea' }}
                                    thumbColor={isPasswordUpdate ? '#ffffff' : '#f1f5f9'}
                                    style={styles.switch}
                                />
                            </View>

                            {isPasswordUpdate && (
                                <Animated.View
                                    style={[
                                        styles.passwordSection,
                                        {
                                            opacity: fadeAnim,
                                            transform: [{ translateY: slideAnim }]
                                        }
                                    ]}
                                >
                                    <View style={styles.sectionHeader}>
                                        <Text style={styles.sectionIcon}>🔒</Text>
                                        <Text style={styles.sectionTitle}>Şifre Güncelle</Text>
                                    </View>
                                    {renderInput('Eski Şifre', 'oldPassword', 'Mevcut şifrenizi girin', true, '🔓')}
                                    {renderInput('Yeni Şifre', 'newPassword', 'Yeni şifrenizi girin', true, '🔐')}
                                    {renderInput('Yeni Şifre (Tekrar)', 'confirmPassword', 'Yeni şifreyi tekrar girin', true, '🔐')}
                                </Animated.View>
                            )}

                            <TouchableOpacity
                                style={[styles.updateButton, loading && styles.updateButtonDisabled]}
                                onPress={handleUpdate}
                                disabled={loading}
                                activeOpacity={0.8}
                            >
                                <LinearGradient
                                    colors={loading ? ['#94a3b8', '#94a3b8'] : ['#10b981', '#059669']}
                                    style={styles.updateButtonGradient}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 0 }}
                                >
                                    {loading ? (
                                        <View style={styles.loadingButtonContent}>
                                            <ActivityIndicator color="#fff" size="small" />
                                            <Text style={styles.updateButtonText}>Güncelleniyor...</Text>
                                        </View>
                                    ) : (
                                        <View style={styles.updateButtonContent}>
                                            <Text style={styles.updateButtonText}>Güncelle</Text>
                                            <Text style={styles.updateButtonIcon}>✨</Text>
                                        </View>
                                    )}
                                </LinearGradient>
                            </TouchableOpacity>
                        </View>
                    </Animated.View>
                </ScrollView>
            </KeyboardAvoidingView>
        </LinearGradient>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    decorativeCircle1: {
        position: 'absolute',
        top: -50,
        right: -50,
        width: 150,
        height: 150,
        borderRadius: 75,
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        opacity: 0.5,
    },
    decorativeCircle2: {
        position: 'absolute',
        bottom: -30,
        left: -30,
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: 'rgba(255, 255, 255, 0.08)',
        opacity: 0.6,
    },
    decorativeCircle3: {
        position: 'absolute',
        top: height * 0.3,
        left: -20,
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: 'rgba(255, 255, 255, 0.06)',
        opacity: 0.7,
    },
    keyboardContainer: {
        flex: 1,
        paddingTop: StatusBar.currentHeight || 0,
    },
    scrollContainer: {
        flexGrow: 1,
        justifyContent: 'center',
        padding: 20,
        paddingTop: 40,
    },
    formContainer: {
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: 24,
        padding: 32,
        width: '100%',
        maxWidth: 500,
        alignSelf: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 20 },
        shadowOpacity: 0.15,
        shadowRadius: 30,
        elevation: 20,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.3)',
        backdropFilter: 'blur(20px)',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    loadingCard: {
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: 20,
        padding: 40,
        alignItems: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.1,
        shadowRadius: 20,
        elevation: 10,
    },
    loadingText: {
        marginTop: 16,
        fontSize: 16,
        color: '#475569',
        fontWeight: '500',
    },
    loadingDots: {
        flexDirection: 'row',
        marginTop: 16,
        gap: 8,
    },
    dot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#667eea',
    },
    dot1: {
        opacity: 0.4,
    },
    dot2: {
        opacity: 0.7,
    },
    dot3: {
        opacity: 1,
    },
    header: {
        alignItems: 'center',
        marginBottom: 32,
    },
    headerIcon: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#667eea',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 16,
        shadowColor: '#667eea',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
        elevation: 8,
    },
    headerIconText: {
        fontSize: 32,
        color: '#fff',
    },
    headerTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: '#1e293b',
        marginBottom: 8,
        textAlign: 'center',
    },
    headerSubtitle: {
        fontSize: 16,
        color: '#64748b',
        textAlign: 'center',
        fontWeight: '400',
    },
    form: {
        backgroundColor: 'transparent',
    },
    inputContainer: {
        marginBottom: 24,
    },
    label: {
        fontSize: 16,
        fontWeight: '600',
        color: '#374151',
        marginBottom: 8,
        marginLeft: 4,
    },
    inputWrapper: {
        position: 'relative',
    },
    inputIcon: {
        position: 'absolute',
        left: 16,
        top: 16,
        fontSize: 20,
        zIndex: 1,
    },
    input: {
        borderWidth: 2,
        borderColor: '#e2e8f0',
        borderRadius: 16,
        paddingHorizontal: 16,
        paddingVertical: 16,
        fontSize: 16,
        backgroundColor: '#f8fafc',
        color: '#1e293b',
        fontWeight: '500',
    },
    inputWithIcon: {
        paddingLeft: 52,
    },
    inputError: {
        borderColor: '#ef4444',
        backgroundColor: '#fef2f2',
    },
    errorContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 8,
        marginLeft: 4,
    },
    errorIcon: {
        fontSize: 12,
        marginRight: 4,
    },
    errorText: {
        color: '#ef4444',
        fontSize: 14,
        fontWeight: '500',
    },
    switchContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginVertical: 24,
        backgroundColor: '#f1f5f9',
        padding: 20,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: '#e2e8f0',
    },
    switchLabelContainer: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    switchIcon: {
        fontSize: 20,
        marginRight: 12,
    },
    switchLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#374151',
    },
    switch: {
        transform: [{ scaleX: 1.2 }, { scaleY: 1.2 }],
    },
    passwordSection: {
        backgroundColor: '#f8fafc',
        padding: 24,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: '#e2e8f0',
        marginBottom: 24,
        shadowColor: '#667eea',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 8,
        elevation: 4,
    },
    sectionHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 16,
    },
    sectionIcon: {
        fontSize: 20,
        marginRight: 12,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#1e293b',
    },
    updateButton: {
        borderRadius: 16,
        marginTop: 8,
        shadowColor: '#10b981',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
        elevation: 8,
    },
    updateButtonDisabled: {
        shadowOpacity: 0.1,
    },
    updateButtonGradient: {
        paddingVertical: 18,
        paddingHorizontal: 32,
        borderRadius: 16,
        alignItems: 'center',
        justifyContent: 'center',
    },
    updateButtonContent: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
    updateButtonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '700',
        marginHorizontal: 8,
    },
    updateButtonIcon: {
        fontSize: 16,
    },
    loadingButtonContent: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
    },
});

export default StudentInformation;