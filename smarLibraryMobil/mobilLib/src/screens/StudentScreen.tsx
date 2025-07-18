import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    TouchableOpacity,
    Alert,
    SafeAreaView,
    StatusBar,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '../contexts/AuthContext';
import { useNavigation } from '@react-navigation/native';

const StudentScreen: React.FC = () => {
    const { user: student, logout } = useAuth();
    const [isLoading, setIsLoading] = useState(false);
    const navigation = useNavigation<any>();

    const handleLogout = async () => {
        try {
            await logout();
        } catch (error) {
            console.error('Çıkış hatası:', error);
            Alert.alert('Hata', 'Çıkış yapılırken bir hata oluştu.');
        }
    };

    if (isLoading) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.loadingContainer}>
                    <Text style={styles.loadingText}>Yükleniyor...</Text>
                </View>
            </SafeAreaView>
        );
    }

    if (!student) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>Öğrenci bilgisi bulunamadı.</Text>
                    <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
                        <Text style={styles.logoutButtonText}>Giriş Sayfasına Dön</Text>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        );
    }

    const infoItems = [
        { label: 'Kullanıcı Adı', value: student.username },
        { label: 'Email', value: student.email },
        { label: 'Telefon', value: student.phone },
    ];

    return (
        <SafeAreaView style={styles.container}>
            <StatusBar barStyle="light-content" backgroundColor="#667eea" />

            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.gradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            >
                <ScrollView
                    contentContainerStyle={styles.scrollContainer}
                    showsVerticalScrollIndicator={false}
                >
                    <View style={styles.card}>
                        <Text style={styles.title}>
                            Hoşgeldiniz {student.fullName}
                        </Text>

                        <View style={styles.infoContainer}>
                            {infoItems.map((item, index) => (
                                <View key={index} style={styles.infoItem}>
                                    <View style={styles.bullet} />
                                    <Text style={styles.infoText}>
                                        <Text style={styles.label}>{item.label}:</Text> {item.value}
                                    </Text>
                                </View>
                            ))}
                        </View>

                        {/* Yeşil navigasyon butonları kaldırıldı */}

                        <TouchableOpacity
                            style={styles.logoutButton}
                            onPress={handleLogout}
                        >
                            <Text style={styles.logoutButtonText}>Çıkış Yap</Text>
                        </TouchableOpacity>
                    </View>
                </ScrollView>
            </LinearGradient>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#667eea' },
    gradient: { flex: 1 },
    scrollContainer: {
        flexGrow: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingVertical: 60,
        paddingHorizontal: 20,
    },
    card: {
        width: '100%',
        maxWidth: 450,
        minHeight: 550,
        backgroundColor: '#fff',
        paddingVertical: 50,
        paddingHorizontal: 35,
        borderRadius: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2,
        shadowRadius: 20,
        elevation: 10,
        alignItems: 'center',
    },
    title: {
        fontSize: 32,
        marginBottom: 40,
        fontWeight: '800',
        borderBottomWidth: 3,
        borderBottomColor: '#764ba2',
        paddingBottom: 15,
        color: '#4b2c91',
        textAlign: 'center',
    },
    infoContainer: {
        width: '100%',
        marginBottom: 30,
    },
    infoItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 20,
        paddingLeft: 30,
        width: '100%',
    },
    bullet: {
        width: 18,
        height: 18,
        backgroundColor: '#764ba2',
        borderRadius: 9,
        position: 'absolute',
        left: 0,
    },
    infoText: {
        fontSize: 18,
        color: '#444',
        flexShrink: 1,
    },
    label: {
        fontWeight: 'bold',
        color: '#4b2c91',
    },
    logoutButton: {
        backgroundColor: '#FF3B30',
        paddingHorizontal: 40,
        paddingVertical: 18,
        borderRadius: 12,
        elevation: 3,
        marginTop: 30,
    },
    logoutButtonText: {
        color: 'white',
        fontSize: 18,
        fontWeight: '600',
        textAlign: 'center',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#667eea',
    },
    loadingText: {
        color: '#fff',
        fontSize: 20,
        fontWeight: '500',
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#667eea',
        paddingHorizontal: 20,
    },
    errorText: {
        color: '#fff',
        fontSize: 20,
        marginBottom: 20,
        textAlign: 'center',
    },
    // navButton ve navButtonText stilleri kaldırıldı
});

export default StudentScreen;
