import React, { useEffect, useState, useRef } from 'react';
import {
    View,
    Text,
    ActivityIndicator,
    TouchableOpacity,
    StyleSheet,
    TextInput,
    Alert,
    Animated,
    RefreshControl,
    ScrollView,
    Dimensions
} from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons } from '@expo/vector-icons';
import { useAuth } from '../contexts/AuthContext';
import {BookRecommendationDTO, ModelStatsDTO, RecommendationResponseDTO} from '../types/book';
import { getBookRecommendations } from '../services/bookService'; // Yeni ekledik
import {
    getAllBooks,
    getBooksByCategory,
    searchBooksByTitle
} from '../services/bookService';
import { BookListDTO } from '../types/book';

// Navigation tipi
type RootStackParamList = {
    BookDetail: { id: number };
};

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'BookDetail'>;

// Kategoriler
const categories = [
    "Tümü",
    "Kişisel Gelişim - Ruhani",
    "İngiliz Edebiyatı",
    "Fransız Edebiyatı",
    "Türk Edebiyatı - Çağdaş",
    "Brezilya Edebiyatı",
    "Distopik - Bilim Kurgu",
    "Türk Edebiyatı - Modern",
    "Türk Edebiyatı - Klasik",
    "Alman Edebiyatı",
    "Ortadoğu Edebiyatı",
    "Amerika Edebiyatı",
    "Latin Amerika Edebiyatı",
    "Distopik - Politik Alegori",
    "Popüler Roman",
    "Avusturya Edebiyatı",
    "Rus Edebiyatı - Klasik",
    "Fantastik - Fantasy",
    "Varoluşçu Edebiyat",
    "Portekiz Edebiyatı"
];


const { width } = Dimensions.get('window');
const ITEMS_PER_PAGE = 6; // Sayfa başına gösterilecek kitap sayısı

const BookListScreen: React.FC = () => {
    const {
        user,
        isAuthenticated,
        favorites,
        addToFavorites,
        removeFromFavorites
    } = useAuth();
    const [showRecommendationPanel, setShowRecommendationPanel] = useState<boolean>(false);
    const [recommendedBook, setRecommendedBook] = useState<BookListDTO | null>(null);
    const [modelStats, setModelStats] = useState<ModelStatsDTO | null>(null);
    const [allBooks, setAllBooks] = useState<BookListDTO[]>([]);
    const [displayedBooks, setDisplayedBooks] = useState<BookListDTO[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [refreshing, setRefreshing] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [searchText, setSearchText] = useState<string>('');
    const [selectedCategory, setSelectedCategory] = useState<string>('Tümü');
    const [currentPage, setCurrentPage] = useState<number>(1);
    const [totalPages, setTotalPages] = useState<number>(1);

    const navigation = useNavigation<NavigationProp>();
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const scaleAnim = useRef(new Animated.Value(0.8)).current;
    const searchTimeoutRef = useRef<number | null>(null);
    const [recommendedBooks, setRecommendedBooks] = useState<BookRecommendationDTO[]>([]);
    const [targetBookTitle, setTargetBookTitle] = useState<string>('');
    const [recommendationLoading, setRecommendationLoading] = useState<boolean>(false);
    const getBookRecommendationsFromAPI = async (addedBook: BookListDTO): Promise<void> => {
        try {
            setRecommendationLoading(true);

            const response: RecommendationResponseDTO = await getBookRecommendations(
                addedBook.title,
                3 // 3 öneri getir
            );

            if (response.success && response.recommendations && response.recommendations.length > 0) {
                setRecommendedBooks(response.recommendations);
                setTargetBookTitle(addedBook.title);
                setShowRecommendationPanel(true);

                // 8 saniye sonra paneli otomatik kapat
                setTimeout(() => {
                    setShowRecommendationPanel(false);
                }, 8000);
            }
        } catch (error) {
            console.error('API önerisi alınamadı:', error);
        } finally {
            setRecommendationLoading(false);
        }
    };


    useEffect(() => {
        fetchBooks();
    }, []);

    useEffect(() => {
        if (searchTimeoutRef.current) {
            clearTimeout(searchTimeoutRef.current);
        }

        searchTimeoutRef.current = setTimeout(() => {
            setCurrentPage(1);
            filterBooks();
        }, 1000) as number;

        return () => {
            if (searchTimeoutRef.current) {
                clearTimeout(searchTimeoutRef.current);
            }
        };
    }, [searchText, selectedCategory]);

    useEffect(() => {
        updateDisplayedBooks();
    }, [allBooks, currentPage]);

    const updateDisplayedBooks = () => {
        const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
        const endIndex = startIndex + ITEMS_PER_PAGE;
        const booksToShow = allBooks.slice(startIndex, endIndex);

        setDisplayedBooks(booksToShow);
        setTotalPages(Math.ceil(allBooks.length / ITEMS_PER_PAGE));
        startAnimations();
    };

    const fetchBooks = async () => {
        setLoading(true);
        setError(null);

        try {
            const data: BookListDTO[] = await getAllBooks();
            setAllBooks(data);
        } catch (err) {
            setError('Kitaplar yüklenirken bir hata oluştu. Lütfen daha sonra tekrar deneyin.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const filterBooks = async () => {
        try {
            setLoading(true);
            let data: BookListDTO[] = [];

            if (searchText.trim()) {
                data = await searchBooksByTitle(searchText);
            } else if (selectedCategory !== 'Tümü') {
                data = await getBooksByCategory(selectedCategory);
            } else {
                data = await getAllBooks();
            }

            setAllBooks(data);
        } catch (err) {
            setError('Kitaplar filtrelenirken hata oluştu');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const startAnimations = () => {
        fadeAnim.setValue(0);
        scaleAnim.setValue(0.8);

        Animated.parallel([
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 600,
                useNativeDriver: true,
            }),
            Animated.spring(scaleAnim, {
                toValue: 1,
                tension: 100,
                friction: 8,
                useNativeDriver: true,
            })
        ]).start();
    };

    const onRefresh = async () => {
        setRefreshing(true);
        setCurrentPage(1);
        await fetchBooks();
        setRefreshing(false);
    };

    const toggleFavorite = async (bookId: number) => {
        if (!isAuthenticated || !user?.id) {
            Alert.alert('Giriş Gerekli', 'Favorilere eklemek için lütfen giriş yapınız');
            return;
        }

        const isFavorite = favorites.has(bookId);

        try {
            if (isFavorite) {
                const success = await removeFromFavorites(bookId);
                if (!success) {
                    Alert.alert('Hata', 'Favorilerden çıkarma işlemi başarısız');
                }
            } else {
                const success = await addToFavorites(bookId);
                if (success) {
                    // Favoriye eklenen kitabı bul
                    const addedBook = allBooks.find(book => book.id === bookId);
                    if (addedBook) {
                        // API'den öneri al
                        await getBookRecommendationsFromAPI(addedBook);
                    }
                } else {
                    Alert.alert('Hata', 'Favorilere ekleme işlemi başarısız');
                }
            }
        } catch (error) {
            console.error('Favori işlemi başarısız:', error);
            Alert.alert('Hata', 'İşlem sırasında bir hata oluştu');
        }
    };

    const goToPage = (page: number) => {
        if (page >= 1 && page <= totalPages) {
            setCurrentPage(page);
        }
    };

    // ... (previous code remains the same)

    const renderBookItem = (item: BookListDTO, index: number) => {
        const isFavorite = favorites.has(item.id);

        return (
            <Animated.View
                key={item.id}
                style={[
                    styles.bookCard,
                    {
                        opacity: fadeAnim,
                        transform: [{ scale: scaleAnim }],
                    }
                ]}
            >
                <TouchableOpacity
                    style={styles.bookContent}
                    onPress={() => navigation.navigate('BookDetail', { id: item.id })}
                    activeOpacity={0.9}
                >
                    <LinearGradient
                        colors={['#667eea', '#764ba2']}
                        style={styles.bookGradient}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                    >
                        <View style={styles.bookIconContainer}>
                            <MaterialIcons name="menu-book" size={40} color="#fff" />
                        </View>
                    </LinearGradient>

                    <View style={styles.bookInfo}>
                        <Text style={styles.bookTitle} numberOfLines={2}>
                            {item.title}
                        </Text>
                        <Text style={styles.bookAuthor} numberOfLines={1}>
                            ✍️ {item.author}
                        </Text>

                        <View style={styles.bookMeta}>
                            <View style={styles.categoryBadge}>
                                <Text style={styles.categoryText} numberOfLines={1}>
                                    {item.category}
                                </Text>
                            </View>

                            <View style={styles.ratingContainer}>
                                <MaterialIcons name="star" size={16} color="#f1c40f" />
                                <Text style={styles.ratingText}>
                                    {item.averageRating ? item.averageRating.toFixed(1) : '0.0'}
                                </Text>
                            </View>
                        </View>
                    </View>

                    <TouchableOpacity
                        onPress={() => toggleFavorite(item.id)}
                        style={styles.favoriteButton}
                    >
                        <MaterialIcons
                            name={isFavorite ? "favorite" : "favorite-outline"}
                            size={24}
                            color={isFavorite ? "#e74c3c" : "#bdc3c7"}
                        />
                    </TouchableOpacity>
                </TouchableOpacity>
            </Animated.View>
        );
    };

// ... (rest of the code remains the same)

    const renderPagination = () => {
        if (totalPages <= 1) return null;

        const pageNumbers = [];
        const maxVisiblePages = 5;
        let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2));
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);

        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }

        for (let i = startPage; i <= endPage; i++) {
            pageNumbers.push(i);
        }

        return (
            <View style={styles.paginationContainer}>
                <TouchableOpacity
                    style={[styles.paginationButton, currentPage === 1 && styles.disabledButton]}
                    onPress={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1}
                >
                    <MaterialIcons name="chevron-left" size={24} color={currentPage === 1 ? "#bdc3c7" : "#fff"} />
                </TouchableOpacity>

                {startPage > 1 && (
                    <>
                        <TouchableOpacity
                            style={styles.pageNumberButton}
                            onPress={() => goToPage(1)}
                        >
                            <Text style={styles.pageNumberText}>1</Text>
                        </TouchableOpacity>
                        {startPage > 2 && <Text style={styles.ellipsis}>...</Text>}
                    </>
                )}

                {pageNumbers.map(page => (
                    <TouchableOpacity
                        key={page}
                        style={[
                            styles.pageNumberButton,
                            currentPage === page && styles.activePageButton
                        ]}
                        onPress={() => goToPage(page)}
                    >
                        <Text style={[
                            styles.pageNumberText,
                            currentPage === page && styles.activePageText
                        ]}>
                            {page}
                        </Text>
                    </TouchableOpacity>
                ))}

                {endPage < totalPages && (
                    <>
                        {endPage < totalPages - 1 && <Text style={styles.ellipsis}>...</Text>}
                        <TouchableOpacity
                            style={styles.pageNumberButton}
                            onPress={() => goToPage(totalPages)}
                        >
                            <Text style={styles.pageNumberText}>{totalPages}</Text>
                        </TouchableOpacity>
                    </>
                )}

                <TouchableOpacity
                    style={[styles.paginationButton, currentPage === totalPages && styles.disabledButton]}
                    onPress={() => goToPage(currentPage + 1)}
                    disabled={currentPage === totalPages}
                >
                    <MaterialIcons name="chevron-right" size={24} color={currentPage === totalPages ? "#bdc3c7" : "#fff"} />
                </TouchableOpacity>
            </View>
        );
    };

    const renderHeader = () => (
        <View style={styles.headerContainer}>
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.headerGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            >
                <Text style={styles.headerTitle}>📚 Kitap Koleksiyonu</Text>
                {isAuthenticated && user && (
                    <Text style={styles.welcomeText}>
                        Hoş geldin, <Text style={styles.userName}>{user.fullName}</Text>
                    </Text>
                )}
            </LinearGradient>

            <View style={styles.controlsContainer}>
                <View style={styles.searchContainer}>
                    <LinearGradient
                        colors={['#f8f9fa', '#e9ecef']}
                        style={styles.searchGradient}
                    >
                        <MaterialIcons name="search" size={24} color="#6c757d" />
                        <TextInput
                            style={styles.searchInput}
                            placeholder="Kitap ara..."
                            placeholderTextColor="#6c757d"
                            value={searchText}
                            onChangeText={setSearchText}
                        />
                        {searchText.length > 0 && (
                            <TouchableOpacity onPress={() => setSearchText('')}>
                                <MaterialIcons name="clear" size={20} color="#6c757d" />
                            </TouchableOpacity>
                        )}
                    </LinearGradient>
                </View>

                <View style={styles.pickerContainer}>
                    <Text style={styles.pickerLabel}>Kategori</Text>
                    <LinearGradient
                        colors={['#f8f9fa', '#e9ecef']}
                        style={styles.pickerGradient}
                    >
                        <Picker
                            selectedValue={selectedCategory}
                            onValueChange={setSelectedCategory}
                            style={styles.picker}
                        >
                            {categories.map((category) => (
                                <Picker.Item key={category} label={category} value={category} />
                            ))}
                        </Picker>
                    </LinearGradient>
                </View>
            </View>
        </View>
    );

    const renderRecommendationPanel = () => {
        if (!showRecommendationPanel || recommendedBooks.length === 0) return null;

        return (
            <Animated.View
                style={[
                    styles.recommendationPanel,
                    {
                        opacity: fadeAnim,
                        transform: [{ translateY: Animated.multiply(fadeAnim, -20) }]
                    }
                ]}
            >
                <LinearGradient
                    colors={['#ff9a56', '#ff6b6b']}
                    style={styles.recommendationGradient}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                >
                    <TouchableOpacity
                        style={styles.closeRecommendation}
                        onPress={() => setShowRecommendationPanel(false)}
                    >
                        <MaterialIcons name="close" size={20} color="#fff" />
                    </TouchableOpacity>

                    <View style={styles.recommendationHeader}>
                        <MaterialIcons name="auto-awesome" size={24} color="#fff" />
                        <Text style={styles.recommendationTitle}>
                            &quot;{targetBookTitle} için öneriler
                        </Text>
                    </View>

                    {recommendationLoading ? (
                        <View style={styles.recommendationLoading}>
                            <ActivityIndicator size="small" color="#fff" />
                            <Text style={styles.loadingText}>Öneriler hazırlanıyor...</Text>
                        </View>
                    ) : (
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            style={styles.recommendationsScroll}
                        >
                            {recommendedBooks.map((book, index) => (
                                <View key={index} style={styles.recommendationCard}>
                                    <View style={styles.recommendationBookInfo}>
                                        <MaterialIcons name="menu-book" size={18} color="#fff" />
                                        <View style={styles.recommendationTextContainer}>
                                            <Text style={styles.recommendationBookTitle} numberOfLines={2}>
                                                {book.title}
                                            </Text>
                                            <Text style={styles.recommendationBookAuthor} numberOfLines={1}>
                                                {book.author}
                                            </Text>
                                        </View>
                                    </View>

                                    <View style={styles.recommendationActions}>
                                        <TouchableOpacity
                                            style={styles.viewBookButton}
                                            onPress={() => {
                                                setShowRecommendationPanel(false);
                                                // Kitabı ID ile bul ve detayına git
                                                const foundBook = allBooks.find(b => b.title === book.title);
                                                if (foundBook) {
                                                    navigation.navigate('BookDetail', { id: foundBook.id });
                                                }
                                            }}
                                        >
                                            <Text style={styles.viewBookButtonText}>Görüntüle</Text>
                                        </TouchableOpacity>

                                        <TouchableOpacity
                                            style={styles.addToFavoritesButton}
                                            onPress={() => {
                                                const foundBook = allBooks.find(b => b.title === book.title);
                                                if (foundBook) {
                                                    toggleFavorite(foundBook.id);
                                                }
                                            }}
                                        >
                                            <MaterialIcons name="favorite-outline" size={14} color="#fff" />
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            ))}
                        </ScrollView>
                    )}

                    {modelStats && (
                        <View style={styles.modelStatsInfo}>
                            <Text style={styles.modelStatsText}>
                                🤖 {modelStats.totalBooks} kitap arasından {recommendedBooks.length} öneri
                            </Text>
                        </View>
                    )}
                </LinearGradient>
            </Animated.View>
        );
    };


    const renderStats = () => (
        <View style={styles.statsContainer}>
            <View style={styles.statItem}>
                <MaterialIcons name="library-books" size={20} color="#3498db" />
                <Text style={styles.statText}>Toplam: {allBooks.length}</Text>
            </View>
            <View style={styles.statItem}>
                <MaterialIcons name="visibility" size={20} color="#2ecc71" />
                <Text style={styles.statText}>Sayfa: {currentPage}/{totalPages}</Text>
            </View>
            <View style={styles.statItem}>
                <MaterialIcons name="grid-view" size={20} color="#e74c3c" />
                <Text style={styles.statText}>Gösterilen: {displayedBooks.length}</Text>
            </View>
        </View>
    );

    if (loading && !refreshing) {
        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.loadingContainer}
            >
                <ActivityIndicator size="large" color="#fff" />
                <Text style={styles.loadingText}>Kitaplar yükleniyor...</Text>
            </LinearGradient>
        );
    }

    return (
        <View style={styles.container}>
            <ScrollView
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl
                        refreshing={refreshing}
                        onRefresh={onRefresh}
                        colors={['#667eea']}
                        tintColor="#667eea"
                    />
                }
            >
                {renderHeader()}
                {renderRecommendationPanel()}
                {renderStats()}

                {error ? (
                    <View style={styles.errorContainer}>
                        <MaterialIcons name="error-outline" size={48} color="#e74c3c" />
                        <Text style={styles.errorText}>{error}</Text>
                    </View>
                ) : displayedBooks.length === 0 ? (
                    <View style={styles.emptyContainer}>
                        <MaterialIcons name="menu-book" size={80} color="#bdc3c7" />
                        <Text style={styles.emptyTitle}>Kitap bulunamadı</Text>
                        <Text style={styles.emptyText}>Farklı bir kategori veya arama terimi deneyin</Text>
                    </View>
                ) : (
                    <>
                        <View style={styles.booksGrid}>
                            {displayedBooks.map((book, index) => renderBookItem(book, index))}
                        </View>
                        {renderPagination()}
                    </>
                )}
            </ScrollView>
        </View>
    );
};


const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f1f3f6',
    },
    headerContainer: {
        marginBottom: 20,
    },
    headerGradient: {
        padding: 30,
        borderBottomLeftRadius: 30,
        borderBottomRightRadius: 30,
        alignItems: 'center',
    },
    headerTitle: {
        fontSize: 32,
        fontWeight: '900',
        color: '#fff',
        marginBottom: 8,
        textShadowColor: 'rgba(0,0,0,0.3)',
        textShadowOffset: { width: 2, height: 2 },
        textShadowRadius: 4,
    },
    welcomeText: {
        fontSize: 16,
        color: 'rgba(255,255,255,0.9)',
        textAlign: 'center',
    },
    userName: {
        fontWeight: '700',
        color: '#fff',
    },
    controlsContainer: {
        padding: 20,
        backgroundColor: '#fff',
        marginHorizontal: 15,
        marginTop: -20,
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.1,
        shadowRadius: 20,
        elevation: 10,
    },
    searchContainer: {
        marginBottom: 20,
    },
    searchGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingVertical: 15,
        borderRadius: 15,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.05)',
    },
    searchInput: {
        flex: 1,
        marginLeft: 12,
        fontSize: 16,
        color: '#2c3e50',
    },
    pickerContainer: {
        marginBottom: 10,
    },
    pickerLabel: {
        fontSize: 16,
        fontWeight: '700',
        color: '#2c3e50',
        marginBottom: 10,
    },
    pickerGradient: {
        borderRadius: 15,
        borderWidth: 1,
        borderColor: 'rgba(0,0,0,0.05)',
        overflow: 'hidden',
    },
    picker: {
        height: 50,
        color: '#2c3e50',
    },
    statsContainer: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginHorizontal: 15,
        marginBottom: 20,
        backgroundColor: '#fff',
        borderRadius: 15,
        paddingVertical: 15,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 5,
    },
    statItem: {
        alignItems: 'center',
    },
    statText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#2c3e50',
        marginTop: 5,
    },
    booksGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        paddingHorizontal: 15,
    },
    bookCard: {
        width: (width - 45) / 2,
        backgroundColor: '#fff',
        borderRadius: 20,
        marginBottom: 15,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.12,
        shadowRadius: 20,
        elevation: 8,
    },
    bookContent: {
        padding: 15,
    },
    bookGradient: {
        height: 80,
        borderRadius: 15,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 15,
    },
    bookIconContainer: {
        backgroundColor: 'rgba(255,255,255,0.2)',
        borderRadius: 25,
        padding: 10,
    },
    bookInfo: {
        flex: 1,
    },
    bookTitle: {
        fontSize: 16,
        fontWeight: '800',
        color: '#2c3e50',
        marginBottom: 8,
        lineHeight: 22,
    },
    bookAuthor: {
        fontSize: 13,
        color: '#7f8c8d',
        marginBottom: 12,
    },
    bookMeta: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    categoryBadge: {
        backgroundColor: '#e8f4f8',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
        flex: 1,
        marginRight: 8,
    },
    categoryText: {
        fontSize: 10,
        fontWeight: '600',
        color: '#3498db',
        textAlign: 'center',
    },
    ratingContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#fff9e6',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
    },
    ratingText: {
        fontSize: 12,
        fontWeight: '700',
        color: '#f39c12',
        marginLeft: 4,
    },
    favoriteButton: {
        position: 'absolute',
        top: 10,
        right: 10,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: 20,
        padding: 8,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    paginationContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginHorizontal: 15,
        marginVertical: 20,
        backgroundColor: '#fff',
        borderRadius: 15,
        padding: 15,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 5,
    },
    paginationButton: {
        backgroundColor: '#3498db',
        borderRadius: 12,
        padding: 12,
        marginHorizontal: 5,
    },
    disabledButton: {
        backgroundColor: '#ecf0f1',
    },
    pageNumberButton: {
        backgroundColor: '#ecf0f1',
        borderRadius: 12,
        paddingHorizontal: 15,
        paddingVertical: 12,
        marginHorizontal: 3,
    },
    activePageButton: {
        backgroundColor: '#3498db',
    },
    pageNumberText: {
        fontSize: 14,
        fontWeight: '700',
        color: '#2c3e50',
    },
    activePageText: {
        color: '#fff',
    },
    ellipsis: {
        fontSize: 16,
        color: '#7f8c8d',
        marginHorizontal: 5,
    },
    emptyContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 40,
        backgroundColor: '#fff',
        margin: 15,
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 5,
    },
    emptyTitle: {
        fontSize: 24,
        fontWeight: '800',
        color: '#2c3e50',
        marginTop: 20,
        marginBottom: 10,
    },
    emptyText: {
        fontSize: 16,
        color: '#7f8c8d',
        textAlign: 'center',
        lineHeight: 24,
    },
    errorContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 40,
        backgroundColor: '#fff',
        margin: 15,
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 5 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 5,
    },
    errorText: {
        fontSize: 16,
        color: '#e74c3c',
        textAlign: 'center',
        marginTop: 15,
        fontWeight: '600',
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    loadingText: {
        marginTop: 20,
        color: '#fff',
        fontSize: 18,
        fontWeight: '600',
    },
    recommendationPanel: {
        marginHorizontal: 15,
        marginBottom: 20,
        borderRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
        elevation: 15,
        overflow: 'hidden',
    },
    recommendationGradient: {
        padding: 16,
        borderRadius: 20,
    },
    closeRecommendation: {
        position: 'absolute',
        top: 10,
        right: 10,
        backgroundColor: 'rgba(255,255,255,0.2)',
        borderRadius: 15,
        padding: 5,
        zIndex: 1,
    },
    recommendationHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 12,
    },
    recommendationTitle: {
        fontSize: 15,
        fontWeight: '800',
        color: '#fff',
        marginLeft: 10,
        flex: 1,
    },
    recommendationsScroll: {
        maxHeight: 120,
        paddingVertical: 4,
    },
    recommendationCard: {
        width: (width - 80) / 3, // 3 kart için genişliği hesapla
        backgroundColor: 'rgba(255,255,255,0.15)',
        borderRadius: 12,
        padding: 10, // padding'i azalttık
        marginRight: 8, // margin'i azalttık
    },
    recommendationBookInfo: {
        flexDirection: 'row',
    },
    recommendationTextContainer: {
        flex: 1,
        marginLeft: 10,
    },
    recommendationBookTitle: {
        fontSize: 12, // 13'ten 12'ye düşürdük
        fontWeight: '700',
        color: '#fff',
        marginBottom: 4,
    },
    recommendationBookAuthor: {
        fontSize: 10, // 11'den 10'a düşürdük
        color: 'rgba(255,255,255,0.8)',
        marginBottom: 6,
    },

    recommendationActions: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: 8,
    },
    viewBookButton: {
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: 8, // 12'den 8'e düşürdük
        paddingVertical: 4, // 6'dan 4'e düşürdük
        borderRadius: 12, // 15'ten 12'ye düşürdük
    },

// viewBookButtonText font size'ını azaltın
    viewBookButtonText: {
        color: '#fff',
        fontWeight: '600',
        fontSize: 10, // 11'den 10'a düşürdük
    },
    addToFavoritesButton: {
        backgroundColor: 'rgba(255,255,255,0.3)',
        padding: 4, // 6'dan 4'e düşürdük
        borderRadius: 12, // 15'ten 12'ye düşürdük
        alignItems: 'center',
        justifyContent: 'center',
    },
    recommendationLoading: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
    },
    modelStatsInfo: {
        marginTop: 8,
        alignItems: 'center',
    },
    modelStatsText: {
        fontSize: 11,
        color: 'rgba(255,255,255,0.8)',
        fontStyle: 'italic',
    },
});

export default BookListScreen;