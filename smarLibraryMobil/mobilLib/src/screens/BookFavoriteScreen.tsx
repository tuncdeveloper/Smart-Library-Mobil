import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    ActivityIndicator,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Animated,
    Easing,
    ScrollView
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { BookFavoriteDTO } from '../types/book';
import { useAuth } from '../contexts/AuthContext';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons } from '@expo/vector-icons';
import { removeFavoriteBook } from '../services/bookService';
import { Alert } from 'react-native';

type RootStackParamList = {
    BookDetail: { id: number };
    Books: undefined;
    Login: undefined;
};

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'BookDetail'>;

const FavoriteBooksPage: React.FC = () => {
    const [error, setError] = useState<string | null>(null);
    const heartAnim = new Animated.Value(1);

    const navigation = useNavigation<NavigationProp>();
    const {
        user,
        isAuthenticated,
        favoriteBooks,
        favoritesLoading,
        refreshFavorites
    } = useAuth();

    const handleRemoveFromFavorites = async (bookId: number, bookTitle: string) => {
        if (!user) return;

        try {
            await removeFavoriteBook(user.id, bookId);
            // Context'ten favorileri yenile
            await refreshFavorites();
        } catch (error) {
            console.error('Favorilerden çıkarma hatası:', error);
            Alert.alert('Hata', 'Kitap favorilerden çıkarılırken bir hata oluştu.');
        }
    };

    // Kalp animasyonu
    useEffect(() => {
        const heartAnimation = () => {
            Animated.sequence([
                Animated.timing(heartAnim, {
                    toValue: 1.1,
                    duration: 500,
                    easing: Easing.ease,
                    useNativeDriver: true,
                }),
                Animated.timing(heartAnim, {
                    toValue: 1,
                    duration: 500,
                    easing: Easing.ease,
                    useNativeDriver: true,
                }),
            ]).start(() => heartAnimation());
        };

        heartAnimation();
    }, []);

    // Sayfa açıldığında favorileri yenile (eğer context'te veri yoksa)
    useEffect(() => {
        if (isAuthenticated && user && favoriteBooks.length === 0 && !favoritesLoading) {
            refreshFavorites();
        }
    }, [isAuthenticated, user]);

    const handleBookPress = (bookId: number) => {
        navigation.navigate('BookDetail', { id: bookId });
    };

    const navigateToBooks = () => {
        navigation.navigate('Books');
    };

    const navigateToLogin = () => {
        navigation.navigate('Login');
    };

    // İstatistikler
    const stats = {
        bookCount: favoriteBooks.length,
        categoryCount: new Set(favoriteBooks.map(book => book.category)).size,
        authorCount: new Set(favoriteBooks.map(book => book.author)).size,
    };

    // Kitap kartı bileşeni
    const BookCard = ({ item, index }: { item: BookFavoriteDTO; index: number }) => {
        const [scaleValue] = useState(new Animated.Value(1));
        const [opacityValue] = useState(new Animated.Value(0));
        const [heartScale] = useState(new Animated.Value(1));
        const [isRemoving, setIsRemoving] = useState(false);

        useEffect(() => {
            Animated.timing(opacityValue, {
                toValue: 1,
                duration: 500,
                delay: index * 100,
                easing: Easing.out(Easing.ease),
                useNativeDriver: true,
            }).start();
        }, []);

        const handlePressIn = () => {
            Animated.spring(scaleValue, {
                toValue: 0.98,
                friction: 3,
                useNativeDriver: true,
            }).start();
        };

        const handlePressOut = () => {
            Animated.spring(scaleValue, {
                toValue: 1,
                friction: 3,
                useNativeDriver: true,
            }).start();
        };

        const handleHeartPress = async () => {
            if (isRemoving) return;

            setIsRemoving(true);

            // Kalp animasyonu - önce büyüt sonra küçült
            Animated.sequence([
                Animated.timing(heartScale, {
                    toValue: 1.3,
                    duration: 150,
                    easing: Easing.out(Easing.ease),
                    useNativeDriver: true,
                }),
                Animated.timing(heartScale, {
                    toValue: 0.8,
                    duration: 150,
                    easing: Easing.in(Easing.ease),
                    useNativeDriver: true,
                }),
            ]).start();

            // Kart animasyonu - soldur ve küçült
            Animated.parallel([
                Animated.timing(opacityValue, {
                    toValue: 0,
                    duration: 300,
                    easing: Easing.in(Easing.ease),
                    useNativeDriver: true,
                }),
                Animated.timing(scaleValue, {
                    toValue: 0.8,
                    duration: 300,
                    easing: Easing.in(Easing.ease),
                    useNativeDriver: true,
                }),
            ]).start();

            // API çağrısını yap
            await handleRemoveFromFavorites(item.bookId, item.title);

            setIsRemoving(false);
        };

        return (
            <Animated.View
                style={[
                    styles.bookCard,
                    {
                        opacity: opacityValue,
                        transform: [{ scale: scaleValue }]
                    }
                ]}
            >
                {/* Kalp butonu - sağ üst köşe */}
                <TouchableOpacity
                    style={styles.heartButton}
                    onPress={handleHeartPress}
                    disabled={isRemoving}
                    activeOpacity={0.7}
                >
                    <Animated.View style={{ transform: [{ scale: heartScale }] }}>
                        {isRemoving ? (
                            <ActivityIndicator size="small" color="#e74c3c" />
                        ) : (
                            <MaterialIcons name="favorite" size={24} color="#e74c3c" />
                        )}
                    </Animated.View>
                </TouchableOpacity>

                <TouchableOpacity
                    activeOpacity={0.9}
                    onPressIn={handlePressIn}
                    onPressOut={handlePressOut}
                    onPress={() => handleBookPress(item.bookId)}
                    style={styles.bookCardContent}
                >
                    <Text style={styles.bookCardTitle}>📖 {item.title}</Text>
                    <Text style={styles.bookCardInfo}>✍️ Yazar: {item.author}</Text>
                    <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                        <Text style={styles.bookCardInfo}>📂 Kategori: </Text>
                        <Text style={styles.categoryBadge}>{item.category}</Text>
                    </View>
                    <View style={styles.bookCardHint}>
                        <Text style={styles.bookCardHintText}>👆 Detayları görüntülemek için tıklayın</Text>
                    </View>
                </TouchableOpacity>
            </Animated.View>
        );
    };

    if (favoritesLoading) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.container}
            >
                <View style={styles.content}>
                    <View style={styles.loadingContainer}>
                        <ActivityIndicator size="large" color="#fff" />
                        <Text style={styles.loadingText}>Favorileriniz yükleniyor...</Text>
                    </View>
                </View>
            </LinearGradient>
        );
    }

    if (!isAuthenticated || !user) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.container}
            >
                <View style={styles.content}>
                    <View style={styles.errorContainer}>
                        <MaterialIcons name="error-outline" size={40} color="#fff" />
                        <Text style={styles.errorText}>Favori kitaplarınızı görüntülemek için giriş yapmalısınız.</Text>
                        <TouchableOpacity
                            style={[styles.button, { backgroundColor: '#27ae60' }]}
                            onPress={navigateToLogin}
                        >
                            <Text style={styles.buttonText}>🚀 Giriş Yap</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </LinearGradient>
        );
    }

    if (error) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.container}
            >
                <View style={styles.content}>
                    <View style={styles.errorContainer}>
                        <MaterialIcons name="error" size={40} color="#fff" />
                        <Text style={styles.errorText}>{error}</Text>
                        <TouchableOpacity
                            style={[styles.button, { backgroundColor: '#667eea' }]}
                        >
                            <Text style={styles.buttonText}>🔄 Yeniden Dene</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </LinearGradient>
        );
    }

    if (favoriteBooks.length === 0) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.container}
            >
                <ScrollView contentContainerStyle={styles.scrollContent}>
                    <View style={styles.content}>
                        <Animated.View style={[styles.titleContainer, { transform: [{ scale: heartAnim }] }]}>
                            <Text style={styles.title}>💔 Favorilerim</Text>
                        </Animated.View>

                        <View style={styles.emptyContainer}>
                            <Animated.Text
                                style={[styles.emptyIcon, { transform: [{ translateY: heartAnim.interpolate({
                                            inputRange: [1, 1.1],
                                            outputRange: [0, -10]
                                        }) }] }]}
                            >
                                📚
                            </Animated.Text>
                            <Text style={styles.emptyTitle}>Henüz favori kitabınız bulunmamaktadır</Text>
                            <Text style={styles.emptyMessage}>
                                Kitapları keşfetmek ve favorilerinize eklemek için aşağıdaki butonu kullanabilirsiniz
                            </Text>

                            <TouchableOpacity
                                style={[styles.button, { backgroundColor: '#27ae60' }]}
                                onPress={navigateToBooks}
                            >
                                <Text style={styles.buttonText}>📖 Kitapları Keşfet</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </ScrollView>
            </LinearGradient>
        );
    }

    return (
        <LinearGradient
            colors={['#667eea', '#764ba2']}
            style={styles.container}
        >
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.content}>
                    <Animated.View style={[styles.titleContainer, { transform: [{ scale: heartAnim }] }]}>
                        <Text style={styles.title}>❤️ Favorilerim</Text>
                    </Animated.View>

                    {/* İstatistikler */}
                    <View style={styles.statsContainer}>
                        <View style={styles.statItem}>
                            <Text style={styles.statNumber}>{stats.bookCount}</Text>
                            <Text style={styles.statLabel}>Favori Kitap</Text>
                        </View>
                        <View style={styles.statItem}>
                            <Text style={styles.statNumber}>{stats.categoryCount}</Text>
                            <Text style={styles.statLabel}>Farklı Kategori</Text>
                        </View>
                        <View style={styles.statItem}>
                            <Text style={styles.statNumber}>{stats.authorCount}</Text>
                            <Text style={styles.statLabel}>Farklı Yazar</Text>
                        </View>
                    </View>

                    {/* Kitap Listesi */}
                    <View style={styles.bookGrid}>
                        {favoriteBooks.map((item, index) => (
                            <BookCard key={item.bookId} item={item} index={index} />
                        ))}
                    </View>

                    <TouchableOpacity
                        style={[styles.button, { marginTop: 20 }]}
                        onPress={navigateToBooks}
                    >
                        <Text style={styles.buttonText}>📚 Daha Fazla Kitap Keşfet</Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </LinearGradient>
    );
};

const { width } = Dimensions.get('window');
const CARD_WIDTH = (width - 48) / 2; // 16*3 padding

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollContent: {
        flexGrow: 1,
        padding: 16,
    },
    content: {
        flex: 1,
        paddingBottom: 20,
    },
    titleContainer: {
        alignItems: 'center',
        marginBottom: 30,
    },
    title: {
        fontSize: 32,
        fontWeight: '700',
        color: '#fff',
        textAlign: 'center',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 40,
    },
    loadingText: {
        color: '#fff',
        fontSize: 18,
        marginTop: 20,
    },
    errorContainer: {
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    errorText: {
        color: '#fff',
        fontSize: 18,
        textAlign: 'center',
        marginVertical: 20,
    },
    button: {
        backgroundColor: '#667eea',
        borderRadius: 10,
        padding: 15,
        alignItems: 'center',
        justifyContent: 'center',
        marginTop: 10,
        flexDirection: 'row',
    },
    buttonText: {
        color: '#fff',
        fontSize: 18,
        fontWeight: '500',
    },
    emptyContainer: {
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        borderRadius: 15,
        padding: 40,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    emptyIcon: {
        fontSize: 64,
        marginBottom: 20,
    },
    emptyTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: '#fff',
        marginBottom: 10,
        textAlign: 'center',
    },
    emptyMessage: {
        fontSize: 16,
        color: '#e0e0e0',
        textAlign: 'center',
        marginBottom: 20,
    },
    statsContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        borderRadius: 15,
        padding: 20,
        marginBottom: 20,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    statItem: {
        alignItems: 'center',
        flex: 1,
    },
    statNumber: {
        fontSize: 28,
        fontWeight: '700',
        color: '#fff',
    },
    statLabel: {
        fontSize: 14,
        color: '#e0e0e0',
        marginTop: 5,
        textAlign: 'center',
    },
    bookGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
    },
    heartButton: {
        position: 'absolute',
        top: 10,
        right: 10,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: 20,
        width: 40,
        height: 40,
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1,
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.15,
        shadowRadius: 4,
        elevation: 4,
    },
    bookCardContent: {
        flex: 1,
    },
    bookCard: {
        width: CARD_WIDTH,
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        borderRadius: 16,
        padding: 16,
        paddingTop: 20,
        marginBottom: 16,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 15,
        elevation: 5,
        position: 'relative',
    },
    bookCardTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: '#2c3e50',
        marginBottom: 8,
    },
    bookCardInfo: {
        fontSize: 13,
        color: '#7f8c8d',
        marginBottom: 4,
    },
    categoryBadge: {
        backgroundColor: '#e8f4f8',
        color: '#2c3e50',
        borderRadius: 20,
        paddingHorizontal: 8,
        paddingVertical: 2,
        fontSize: 12,
        fontWeight: '500',
        overflow: 'hidden',
    },
    bookCardHint: {
        marginTop: 10,
        padding: 8,
        backgroundColor: 'rgba(102, 126, 234, 0.1)',
        borderRadius: 8,
    },
    bookCardHintText: {
        fontSize: 12,
        color: '#667eea',
        textAlign: 'center',
    },
});

export default FavoriteBooksPage;