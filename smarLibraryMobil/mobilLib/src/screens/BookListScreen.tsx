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
    "Psikolojik",
    "Aşk",
    "Tarihi Roman",
    "Macera",
    "Toplumcu Gerçekçi",
    "Felsefi Roman",
    "Bilim Kurgu (Sci-Fi)",
    "Siyasi Alegori",
    "Fantastik",
    "Varoluşçu",
    "Modern",
    "Toplumsal Roman",
    "Dram"
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
                if (!success) {
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
                                    {item.averageRating?.toFixed(1) || '0.0'}
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
});

export default BookListScreen;