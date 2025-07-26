    import React, { useEffect, useState } from 'react';
    import {
        View,
        Text,
        StyleSheet,
        ScrollView,
        ActivityIndicator,
        TextInput,
        TouchableOpacity,
        Alert,
        Dimensions
    } from 'react-native';
    import { LinearGradient } from 'expo-linear-gradient';
    import { RouteProp, useRoute, useNavigation } from '@react-navigation/native';
    import type { StackNavigationProp } from '@react-navigation/stack';
    import {
        getBookById,
        commentBook,
        getBookComments,
        rateBook,
        getAverageRating,
        getUserRating,
        addFavoriteBook,
        removeFavoriteBook,
        getFavoriteBooksByStudentId,
        getBookRecommendations,
        searchBooksByTitle
    } from '../services/bookService';
    import {
        BookDetailDTO,
        BookRatingRequestDTO,
        BookRatingResponseDTO,
        BookRecommendationDTO
    } from '../types/book';
    import { useAuth } from '../contexts/AuthContext';

    type RootStackParamList = {
        BookDetails: { id: number };
        HomeScreen: undefined;
        BookListScreen: undefined;
        BookFavoriteScreen: undefined;
    };

    type BookDetailsRouteProp = RouteProp<{ BookDetails: { id: number } }, 'BookDetails'>;
    type BookDetailsNavigationProp = StackNavigationProp<RootStackParamList>;

    const { width } = Dimensions.get('window');

    const BookDetailsScreen: React.FC = () => {
        const route = useRoute<BookDetailsRouteProp>();
        const navigation = useNavigation<BookDetailsNavigationProp>();
        const { id } = route.params;

        const {
            user,
            addToFavorites,
            removeFromFavorites,
            refreshFavorites
        } = useAuth();

        const studentId = user?.id;

        const [book, setBook] = useState<BookDetailDTO | null>(null);
        const [loading, setLoading] = useState<boolean>(true);
        const [userRating, setUserRating] = useState<number>(0);
        const [averageRating, setAverageRating] = useState<number | null>(null);
        const [comment, setComment] = useState<string>('');
        const [comments, setComments] = useState<BookRatingResponseDTO[]>([]);
        const [commentLoading, setCommentLoading] = useState(false);
        const [ratingLoading, setRatingLoading] = useState(false);
        const [userComment, setUserComment] = useState<BookRatingResponseDTO | null>(null);
        const [isEditingComment, setIsEditingComment] = useState(false);
        const [isFavorite, setIsFavorite] = useState<boolean>(false);
        const [favoriteLoading, setFavoriteLoading] = useState<boolean>(false);
        const [recommendations, setRecommendations] = useState<BookRecommendationDTO[]>([]);

        useEffect(() => {
            const fetchBook = async () => {
                setLoading(true);
                try {
                    const data = await getBookById(id);
                    setBook(data);

                    if (data?.title) {
                        await fetchRecommendations(data.title);
                    }

                    const avg = await getAverageRating(id);
                    setAverageRating(avg);

                    if (user && user.id) {
                        const userRate = await getUserRating(id, user.id);
                        setUserRating(userRate || 0);
                    }
                } catch (error) {
                    console.error('Kitap detayları alınamadı:', error);
                } finally {
                    setLoading(false);
                }
            };

            const checkIfFavorite = async () => {
                if (user && user.id) {
                    try {
                        const favorites = await getFavoriteBooksByStudentId(user.id);
                        const isFav = favorites.some(fav => fav.bookId === id);
                        setIsFavorite(isFav);
                    } catch (error) {
                        console.error('Favori kontrol hatası:', error);
                    }
                }
            };

            const fetchComments = async () => {
                try {
                    const allComments = await getBookComments(id);
                    setComments(allComments);

                    if (user && user.id) {
                        const userCommentData = allComments.find(c => c.studentId === user.id);
                        setUserComment(userCommentData || null);

                        if (userCommentData && userCommentData.comment) {
                            setComment(userCommentData.comment);
                        } else {
                            setComment('');
                        }
                    }
                } catch (err) {
                    console.error('Yorumlar alınamadı:', err);
                }
            };

            fetchBook();
            fetchComments();
            checkIfFavorite();
        }, [id, user?.id]);

        const handleRatingChange = async (newRating: number) => {
            if (!studentId) {
                Alert.alert('Hata', 'Puan vermek için giriş yapmalısınız.');
                return;
            }

            setRatingLoading(true);
            try {
                await rateBook({ bookId: id, studentId, rating: newRating, comment: '' });
                setUserRating(newRating);

                const avg = await getAverageRating(id);
                setAverageRating(avg);

                if (userComment) {
                    setComments(prev =>
                        prev.map(c =>
                            c.studentId === studentId
                                ? { ...c, rating: newRating }
                                : c
                        )
                    );
                    setUserComment(prev => prev ? { ...prev, rating: newRating } : null);
                }

            } catch (error: any) {
                Alert.alert('Hata', error.message || 'Puan verme işlemi sırasında hata oluştu');
            } finally {
                setRatingLoading(false);
            }
        };

        const handleBookPress = async (rec: BookRecommendationDTO) => {
            // BookRecommendationDTO'da ID field'i yok, sadece title ve author var
            // Bu yüzden title ile arama yapıyoruz
            try {
                const searchResults = await searchBooksByTitle(rec.title);
                const foundBook = searchResults.find(book =>
                    book.title.toLowerCase() === rec.title.toLowerCase() &&
                    book.author.toLowerCase() === rec.author.toLowerCase()
                );

                if (foundBook && foundBook.id !== id) {
                    navigation.setParams({ id: foundBook.id });
                } else if (!foundBook) {
                    Alert.alert('Bilgi', 'Bu kitabın detaylarına şu anda erişilemiyor.');
                }
            } catch (error) {
                console.error('Arama hatası:', error);
                Alert.alert('Hata', 'Kitap aranırken bir hata oluştu.');
            }
        };

        const handleCommentSubmit = async () => {
            if (!studentId) {
                Alert.alert('Hata', 'Yorum yapmak için giriş yapmalısınız.');
                return;
            }

            if (!userRating || userRating === 0) {
                Alert.alert('Hata', 'Yorum yapmak için önce kitaba puan vermelisiniz.');
                return;
            }

            if (!comment.trim()) {
                Alert.alert('Hata', 'Lütfen yorum metnini giriniz.');
                return;
            }

            setCommentLoading(true);
            try {
                const commentData: BookRatingRequestDTO = {
                    bookId: id,
                    studentId,
                    comment: comment.trim(),
                    rating: userRating
                };

                const newComment = await commentBook(commentData);

                if (userComment) {
                    setComments(prev =>
                        prev.map(c =>
                            c.studentId === studentId
                                ? {
                                    ...c,
                                    comment: comment.trim(),
                                    rating: userRating,
                                    studentUsername: user?.username,
                                    updatedAt: new Date().toISOString()
                                }
                                : c
                        )
                    );
                    setUserComment({
                        ...userComment,
                        comment: comment.trim(),
                        rating: userRating,
                        studentUsername: user?.username,
                    });
                } else {
                    const commentWithRating = {
                        ...newComment,
                        rating: userRating,
                        studentUsername: user?.username
                    };
                    setComments(prev => [commentWithRating, ...prev]);
                    setUserComment(commentWithRating);
                }

                setIsEditingComment(false);
                Alert.alert('Başarılı', userComment ? 'Yorumunuz güncellendi!' : 'Yorumunuz gönderildi!');

            } catch (error: any) {
                Alert.alert('Hata', error.message || 'Yorum işlemi sırasında hata oluştu');
            } finally {
                setCommentLoading(false);
            }
        };

        const handleFavoriteToggle = async () => {
            if (!user || !user.id) {
                Alert.alert('Hata', 'Favori eklemek için giriş yapmalısınız.');
                return;
            }

            const oldFavoriteState = isFavorite;
            setIsFavorite(!oldFavoriteState);
            setFavoriteLoading(true);

            try {
                if (oldFavoriteState) {
                    await removeFromFavorites(id);
                } else {
                    await addToFavorites(id);
                    if (book?.title) await fetchRecommendations(book.title);
                }
            } catch (error) {
                setIsFavorite(oldFavoriteState);
                Alert.alert('Hata', 'İşlem sırasında bir hata oluştu');
            } finally {
                setFavoriteLoading(false);
            }
        };

        // 2. fetchRecommendations fonksiyonunu da güncelle
        const fetchRecommendations = async (bookTitle: string) => {
            try {
                const result = await getBookRecommendations(bookTitle, 4);
                console.log("API'den gelen TAMAM öneri verisi:");
                console.log(JSON.stringify(result, null, 2));

                if (result.success && result.recommendations) {
                    console.log('İlk öneri objesi TAM HALİ:');
                    console.log(JSON.stringify(result.recommendations[0], null, 2));

                    // Her önerinin field'larını kontrol et
                    result.recommendations.forEach((rec, index) => {
                        console.log(`\n=== ÖNERİ ${index + 1} ===`);
                        console.log('Keys:', Object.keys(rec));
                        console.log('similarity_score var mı?', 'similarity_score' in rec);
                        console.log('similarityScore var mı?', 'similarityScore' in rec);
                        console.log('final_score var mı?', 'final_score' in rec);
                        console.log('finalScore var mı?', 'finalScore' in rec);
                    });

                    setRecommendations(result.recommendations);
                }
            } catch (error) {
                console.error('Öneri alma hatası:', error);
            }
        };

        const handleEditComment = () => {
            setIsEditingComment(true);
            if (userComment && userComment.comment) {
                setComment(userComment.comment);
            }
        };

        const handleCancelEdit = () => {
            setIsEditingComment(false);
            if (userComment && userComment.comment) {
                setComment(userComment.comment);
            } else {
                setComment('');
            }
        };

        const formatDate = (dateString: string) => {
            return new Date(dateString).toLocaleDateString('tr-TR', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        };

        const getUserDisplayName = (comment: BookRatingResponseDTO) => {
            if (comment.studentUsername) {
                return comment.studentUsername;
            }

            if (user && comment.studentId === user.id) {
                return user.username || 'Sen';
            }

            return `Kullanıcı #${comment.studentId}`;
        };

        const renderStars = (rating: number, interactive: boolean = false) => {
            return (
                <View style={styles.starsContainer}>
                    {[1, 2, 3, 4, 5].map((star) => (
                        <TouchableOpacity
                            key={star}
                            onPress={() => interactive && user && handleRatingChange(star)}
                            disabled={!interactive || !user}
                            style={styles.starButton}
                        >
                            <Text style={[
                                styles.star,
                                { color: star <= rating ? '#ffc107' : '#e0e0e0' }
                            ]}>
                                ★
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>
            );
        };

        // Similarity değerini güvenli şekilde formatlayan yardımcı fonksiyon
        // 1. İlk olarak objenin tam içeriğini görelim
        const formatSimilarity = (rec: BookRecommendationDTO) => {
            console.log('=== FULL OBJECT DEBUG ===');
            console.log('rec tam objesi:', JSON.stringify(rec, null, 2));
            console.log('rec.similarity_score:', rec.similarityScore);
            console.log('rec.final_score:', rec.finalScore);

            // Objedeki TÜM key'leri listele
            console.log('Objedeki tüm key\'ler:', Object.keys(rec));

            // Her key'in değerini yazdır
            Object.keys(rec).forEach(key => {
                console.log(`${key}:`, rec[key as keyof BookRecommendationDTO], `(tip: ${typeof rec[key as keyof BookRecommendationDTO]})`);
            });

            const similarity = rec.similarityScore || rec.finalScore || 0;
            console.log('formatSimilarity - seçilen similarity:', similarity);

            if (typeof similarity !== 'number' || isNaN(similarity)) {
                console.log('formatSimilarity - similarity geçersiz, 0 döndürülüyor');
                return 0;
            }

            const result = Math.round(similarity * 100);
            console.log('formatSimilarity - sonuç:', result);
            return result;
        };

        const getSimilarityWidth = (rec: BookRecommendationDTO): number => {
            return formatSimilarity(rec);
        };

    // Alternatif olarak, daha güvenli bir yaklaşım:
        const formatSimilarityAdvanced = (rec: BookRecommendationDTO) => {
            // Hem snake_case hem camelCase desteği
            const similarity = rec.similarityScore || rec.similarityScore ||
                rec.finalScore || rec.finalScore || 0;

            if (typeof similarity !== 'number' || isNaN(similarity)) {
                return 0;
            }

            return Math.round(similarity * 100);
        };
        if (loading) {
            return (
                <LinearGradient
                    colors={['#667eea', '#764ba2']}
                    style={styles.loadingContainer}
                >
                    <View style={styles.loadingContent}>
                        <ActivityIndicator size="large" color="#ffffff" />
                        <Text style={styles.loadingText}>Kitap bilgileri yükleniyor...</Text>
                    </View>
                </LinearGradient>
            );
        }

        if (!book) {
            return (
                <LinearGradient
                    colors={['#667eea', '#764ba2']}
                    style={styles.loadingContainer}
                >
                    <View style={styles.errorContent}>
                        <Text style={styles.errorText}>⚠️ Kitap bulunamadı</Text>
                    </View>
                </LinearGradient>
            );
        }

        return (
            <LinearGradient
                colors={['#667eea', '#764ba2']}
                style={styles.gradientContainer}
            >
                <ScrollView contentContainerStyle={styles.container}>
                    <View style={styles.contentCard}>
                        <View style={styles.titleContainer}>
                            <Text style={styles.title}>
                                📖 {book.title}
                            </Text>
                            <TouchableOpacity
                                style={[
                                    styles.favoriteButton,
                                    { backgroundColor: isFavorite ? '#e74c3c' : '#ecf0f1' }
                                ]}
                                onPress={handleFavoriteToggle}
                                disabled={favoriteLoading || !user}
                            >
                                {favoriteLoading ? (
                                    <ActivityIndicator size="small" color="#ffffff" />
                                ) : (
                                    <Text style={[
                                        styles.favoriteIcon,
                                        { color: isFavorite ? '#ffffff' : '#95a5a6' }
                                    ]}>
                                        ♥
                                    </Text>
                                )}
                            </TouchableOpacity>
                        </View>

                        <View style={styles.infoGrid}>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>✍️ Yazar</Text>
                                <Text style={styles.infoValue}>{book.author}</Text>
                            </View>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>🏢 Yayınevi</Text>
                                <Text style={styles.infoValue}>{book.publisher}</Text>
                            </View>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>📅 Yayın Yılı</Text>
                                <Text style={styles.infoValue}>{book.publicationYear}</Text>
                            </View>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>📚 Kategori</Text>
                                <Text style={styles.infoValue}>{book.category}</Text>
                            </View>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>📃 Sayfa Sayısı</Text>
                                <Text style={styles.infoValue}>{book.pageCount}</Text>
                            </View>
                            <View style={styles.infoItem}>
                                <Text style={styles.infoLabel}>🌐 Dil</Text>
                                <Text style={styles.infoValue}>{book.language}</Text>
                            </View>
                        </View>

                        {recommendations.length > 0 && (
                            <View style={styles.recommendationsSection}>
                                <View style={styles.recommendationHeader}>
                                    <Text style={styles.sectionTitle}>
                                        ✨ Size Özel Öneriler
                                    </Text>
                                    <View style={styles.recommendationBadge}>
                                        <Text style={styles.badgeText}>{recommendations.length}</Text>
                                    </View>
                                </View>
                                <ScrollView
                                    style={styles.recommendationGrid}
                                    showsVerticalScrollIndicator={false}
                                    nestedScrollEnabled={true}
                                >
                                    {/* İlk satır - 0 ve 1 indeksli öneriler */}
                                    <View style={styles.recommendationRow}>
                                        {recommendations.slice(0, 2).map((rec, index) => (
                                            <TouchableOpacity
                                                key={index}
                                                style={styles.recommendationCard}
                                                onPress={() => handleBookPress(rec)}
                                                activeOpacity={0.7}
                                            >
                                                <View style={styles.recommendationGradient}>
                                                    <View style={styles.recommendationContent}>
                                                        <Text style={styles.recommendationTitle} numberOfLines={2}>
                                                            {rec.title}
                                                        </Text>
                                                        <Text style={styles.recommendationAuthor} numberOfLines={1}>
                                                            {rec.author}
                                                        </Text>
                                                        <View style={styles.similarityContainer}>
                                                            <View style={styles.similarityBar}>
                                                                <View
                                                                    style={[
                                                                        styles.similarityFill,
                                                                        {
                                                                            width: `${getSimilarityWidth(rec)}%`
                                                                        }
                                                                    ]}
                                                                />
                                                            </View>
                                                        </View>
                                                    </View>
                                                    <View style={styles.recommendationIcon}>
                                                        <Text style={styles.iconText}>📚</Text>
                                                    </View>
                                                </View>
                                            </TouchableOpacity>
                                        ))}
                                    </View>

                                    {/* İkinci satır - 2 ve 3 indeksli öneriler */}
                                    {recommendations.length > 2 && (
                                        <View style={styles.recommendationRow}>
                                            {recommendations.slice(2, 4).map((rec, index) => (
                                                <TouchableOpacity
                                                    key={index + 2}
                                                    style={styles.recommendationCard}
                                                    onPress={() => handleBookPress(rec)}
                                                    activeOpacity={0.7}
                                                >
                                                    <View style={styles.recommendationGradient}>
                                                        <View style={styles.recommendationContent}>
                                                            <Text style={styles.recommendationTitle} numberOfLines={2}>
                                                                {rec.title}
                                                            </Text>
                                                            <Text style={styles.recommendationAuthor} numberOfLines={1}>
                                                                {rec.author}
                                                            </Text>
                                                            <View style={styles.similarityContainer}>
                                                                <View style={styles.similarityBar}>
                                                                    <View
                                                                        style={[
                                                                            styles.similarityFill,
                                                                            {
                                                                                width: `${getSimilarityWidth(rec)}%`
                                                                            }
                                                                        ]}
                                                                    />
                                                                </View>
                                                            </View>
                                                        </View>
                                                        <View style={styles.recommendationIcon}>
                                                            <Text style={styles.iconText}>📚</Text>
                                                        </View>
                                                    </View>
                                                </TouchableOpacity>
                                            ))}
                                        </View>
                                    )}
                                </ScrollView>
                            </View>
                        )}
                        <View style={styles.descriptionSection}>
                            <Text style={styles.descriptionTitle}>📝 Açıklama</Text>
                            <Text style={styles.description}>{book.description}</Text>
                        </View>

                        <View style={styles.ratingSection}>
                            <View style={styles.averageRatingBox}>
                                <Text style={styles.averageRatingValue}>
                                    {averageRating?.toFixed(1) ?? '0.0'}
                                    <Text style={styles.averageRatingMax}>/5</Text>
                                </Text>
                                <Text style={styles.averageRatingLabel}>Ortalama Puan</Text>
                            </View>

                            <View style={styles.userRatingBox}>
                                <Text style={styles.userRatingLabel}>
                                    {user ? (userRating ? 'Puanınız' : 'Puan Verin') : 'Puan vermek için giriş yapın'}
                                </Text>
                                {renderStars(userRating, true)}
                                {ratingLoading && (
                                    <View style={styles.ratingLoading}>
                                        <ActivityIndicator size="small" color="#667eea" />
                                        <Text style={styles.ratingLoadingText}>Kaydediliyor...</Text>
                                    </View>
                                )}
                            </View>
                        </View>

                        <View style={styles.commentsSection}>
                            <Text style={styles.sectionTitle}>
                                💬 Yorumlar ({comments.length})
                            </Text>

                            {user ? (
                                <View style={styles.commentInputSection}>
                                    <Text style={styles.commentInputTitle}>
                                        {userComment && !isEditingComment ? 'Yorumunuz' : 'Yorum Yapın'}
                                    </Text>

                                    {(!userRating || userRating === 0) && (
                                        <View style={styles.warningBox}>
                                            <Text style={styles.warningText}>
                                                <Text style={styles.warningBold}>Dikkat:</Text> Yorum yapmak için önce kitaba puan vermelisiniz.
                                            </Text>
                                        </View>
                                    )}

                                    {userComment && !isEditingComment ? (
                                        <View style={styles.userCommentDisplay}>
                                            <View style={styles.commentHeader}>
                                                <Text style={styles.commentUser}>Sen</Text>
                                                <Text style={styles.commentDate}>
                                                    {userComment.createdAt && formatDate(userComment.createdAt)}
                                                </Text>
                                            </View>
                                            {userComment.rating > 0 && (
                                                <View style={styles.commentRatingDisplay}>
                                                    {renderStars(userComment.rating)}
                                                </View>
                                            )}
                                            <Text style={styles.commentText}>{userComment.comment}</Text>
                                            <TouchableOpacity
                                                style={styles.editButton}
                                                onPress={handleEditComment}
                                            >
                                                <Text style={styles.editButtonText}>✏️ Düzenle</Text>
                                            </TouchableOpacity>
                                        </View>
                                    ) : (
                                        <>
                                            <TextInput
                                                style={[
                                                    styles.commentInput,
                                                    { borderColor: comment ? '#667eea' : '#e0e0e0' }
                                                ]}
                                                placeholder="Kitap hakkında yorumunuzu yazın..."
                                                value={comment}
                                                onChangeText={setComment}
                                                multiline
                                                editable={!commentLoading && (userRating > 0)}
                                            />
                                            <View style={styles.commentButtons}>
                                                <TouchableOpacity
                                                    style={[
                                                        styles.submitButton,
                                                        {
                                                            backgroundColor: (commentLoading || !comment.trim() || (!userRating || userRating === 0))
                                                                ? '#95a5a6' : '#667eea'
                                                        }
                                                    ]}
                                                    onPress={handleCommentSubmit}
                                                    disabled={commentLoading || !comment.trim() || (!userRating || userRating === 0)}
                                                >
                                                    {commentLoading && <ActivityIndicator size="small" color="#ffffff" />}
                                                    <Text style={styles.submitButtonText}>
                                                        {commentLoading ? 'Gönderiliyor...' : (userComment ? 'Güncelle' : 'Yorum Gönder')}
                                                    </Text>
                                                </TouchableOpacity>
                                                {isEditingComment && (
                                                    <TouchableOpacity
                                                        style={styles.cancelButton}
                                                        onPress={handleCancelEdit}
                                                        disabled={commentLoading}
                                                    >
                                                        <Text style={styles.cancelButtonText}>İptal</Text>
                                                    </TouchableOpacity>
                                                )}
                                            </View>
                                        </>
                                    )}
                                </View>
                            ) : (
                                <View style={styles.loginPrompt}>
                                    <Text style={styles.loginPromptText}>
                                        Yorum yapmak için giriş yapmalısınız
                                    </Text>
                                </View>
                            )}

                            {comments.length === 0 ? (
                                <View style={styles.noComments}>
                                    <Text style={styles.noCommentsIcon}>💬</Text>
                                    <Text style={styles.noCommentsText}>Henüz yorum yapılmamış</Text>
                                </View>
                            ) : (
                                <View style={styles.commentsList}>
                                    {comments.map((commentItem) => (
                                        <View key={commentItem.id} style={styles.commentCard}>
                                            <View style={styles.commentHeader}>
                                                <Text style={styles.commentUser}>
                                                    {getUserDisplayName(commentItem)}
                                                </Text>
                                                <Text style={styles.commentDate}>
                                                    {commentItem.createdAt && formatDate(commentItem.createdAt)}
                                                </Text>
                                            </View>
                                            {commentItem.rating > 0 && (
                                                <View style={styles.commentRatingDisplay}>
                                                    {renderStars(commentItem.rating)}
                                                </View>
                                            )}
                                            <Text style={styles.commentText}>
                                                {commentItem.comment}
                                            </Text>
                                        </View>
                                    ))}
                                </View>
                            )}
                        </View>
                    </View>
                </ScrollView>
            </LinearGradient>
        );
    };




    const styles = StyleSheet.create({
        gradientContainer: {
            flex: 1,
        },
        container: {
            padding: 20,
        },
        contentCard: {
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            borderRadius: 24,
            padding: 30,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 10 },
            shadowOpacity: 0.15,
            shadowRadius: 20,
            elevation: 10,
        },
        loadingContainer: {
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
        },
        loadingContent: {
            alignItems: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            padding: 30,
            borderRadius: 24,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 10 },
            shadowOpacity: 0.15,
            shadowRadius: 20,
            elevation: 10,
        },
        loadingText: {
            marginTop: 15,
            fontSize: 18,
            color: '#667eea',
            fontWeight: '500',
        },
        errorContent: {
            alignItems: 'center',
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            padding: 30,
            borderRadius: 24,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 10 },
            shadowOpacity: 0.15,
            shadowRadius: 20,
            elevation: 10,
        },
        errorText: {
            fontSize: 18,
            color: '#e74c3c',
            fontWeight: '500',
            textAlign: 'center',
        },
        title: {
            fontSize: 28,
            fontWeight: '700',
            color: '#2c3e50',
            textAlign: 'center',
            marginBottom: 30,
            letterSpacing: -0.5,
        },
        infoGrid: {
            flexDirection: 'row',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            marginBottom: 30,
        },
        infoItem: {
            width: '48%',
            backgroundColor: 'rgba(102, 126, 234, 0.08)',
            borderRadius: 12,
            padding: 15,
            marginBottom: 10,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.15)',
        },
        infoLabel: {
            fontSize: 14,
            fontWeight: '600',
            color: '#667eea',
            marginBottom: 5,
        },
        infoValue: {
            fontSize: 16,
            fontWeight: '500',
            color: '#2c3e50',
        },
        descriptionSection: {
            backgroundColor: 'rgba(102, 126, 234, 0.05)',
            borderRadius: 16,
            padding: 20,
            marginBottom: 30,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.1)',
        },
        descriptionTitle: {
            fontSize: 18,
            fontWeight: '700',
            color: '#667eea',
            marginBottom: 10,
        },
        description: {
            fontSize: 16,
            lineHeight: 24,
            color: '#2c3e50',
        },
        ratingSection: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: 'rgba(102, 126, 234, 0.08)',
            borderRadius: 16,
            padding: 20,
            marginBottom: 30,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.15)',
        },
        averageRatingBox: {
            alignItems: 'center',
        },
        averageRatingValue: {
            fontSize: 36,
            fontWeight: '800',
            color: '#667eea',
            marginBottom: 5,
        },
        averageRatingMax: {
            fontSize: 18,
            color: '#95a5a6',
        },
        averageRatingLabel: {
            fontSize: 14,
            fontWeight: '600',
            color: '#7f8c8d',
        },
        userRatingBox: {
            alignItems: 'center',
        },
        userRatingLabel: {
            fontSize: 14,
            fontWeight: '600',
            color: '#7f8c8d',
            marginBottom: 10,
        },
        starsContainer: {
            flexDirection: 'row',
            gap: 5,
        },
        starButton: {
            padding: 5,
        },
        star: {
            fontSize: 28,
        },
        ratingLoading: {
            flexDirection: 'row',
            alignItems: 'center',
            marginTop: 10,
        },
        ratingLoadingText: {
            marginLeft: 8,
            fontSize: 14,
            color: '#667eea',
        },
        commentsSection: {
            marginTop: 10,
        },
        sectionTitle: {
            fontSize: 24,
            fontWeight: '700',
            color: '#2c3e50',
            marginBottom: 20,
            paddingBottom: 10,
            borderBottomWidth: 2,
            borderBottomColor: 'rgba(102, 126, 234, 0.3)',
        },
        commentInputSection: {
            backgroundColor: 'rgba(102, 126, 234, 0.05)',
            borderRadius: 16,
            padding: 20,
            marginBottom: 30,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.1)',
        },
        commentInputTitle: {
            fontSize: 18,
            fontWeight: '600',
            color: '#2c3e50',
            marginBottom: 15,
        },
        warningBox: {
            backgroundColor: '#fff3cd',
            borderLeftWidth: 4,
            borderLeftColor: '#ffc107',
            padding: 12,
            borderRadius: 4,
            marginBottom: 15,
        },
        warningText: {
            color: '#856404',
            fontSize: 14,
        },
        recommendationRow: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            marginBottom: 8,
        },
        warningBold: {
            fontWeight: 'bold',
        },
        userCommentDisplay: {
            backgroundColor: '#f8f9ff',
            borderRadius: 12,
            padding: 15,
            marginBottom: 15,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.2)',
        },
        commentHeader: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 10,
        },
        commentUser: {
            fontSize: 16,
            fontWeight: '700',
            color: '#2c3e50',
        },
        commentDate: {
            fontSize: 14,
            color: '#95a5a6',
        },
        commentRatingDisplay: {
            marginBottom: 10,
        },
        commentText: {
            fontSize: 16,
            lineHeight: 22,
            color: '#2c3e50',
            marginBottom: 10,
        },
        editButton: {
            backgroundColor: '#ffc107',
            borderRadius: 8,
            paddingVertical: 8,
            paddingHorizontal: 16,
            alignSelf: 'flex-start',
        },
        editButtonText: {
            color: '#000',
            fontSize: 14,
            fontWeight: '600',
        },
        commentInput: {
            borderWidth: 1,
            borderRadius: 12,
            padding: 15,
            fontSize: 16,
            minHeight: 100,
            textAlignVertical: 'top',
            marginBottom: 15,
            backgroundColor: '#ffffff',
        },
        commentButtons: {
            flexDirection: 'row',
            alignItems: 'center',
        },
        submitButton: {
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: '#667eea',
            borderRadius: 10,
            paddingVertical: 12,
            paddingHorizontal: 25,
            minWidth: 120,
        },
        submitButtonText: {
            color: '#ffffff',
            fontSize: 16,
            fontWeight: '600',
            marginLeft: 5,
        },
        cancelButton: {
            backgroundColor: '#95a5a6',
            borderRadius: 10,
            paddingVertical: 12,
            paddingHorizontal: 25,
            marginLeft: 10,
        },
        cancelButtonText: {
            color: '#ffffff',
            fontSize: 16,
            fontWeight: '600',
        },
        loginPrompt: {
            backgroundColor: 'rgba(102, 126, 234, 0.05)',
            borderRadius: 16,
            padding: 20,
            marginBottom: 30,
            borderWidth: 1,
            borderStyle: 'dashed',
            borderColor: 'rgba(102, 126, 234, 0.3)',
            alignItems: 'center',
        },
        loginPromptText: {
            color: '#667eea',
            fontSize: 16,
            fontWeight: '500',
            textAlign: 'center',
        },
        noComments: {
            backgroundColor: 'rgba(127, 140, 141, 0.1)',
            borderRadius: 16,
            padding: 40,
            alignItems: 'center',
            borderWidth: 2,
            borderStyle: 'dashed',
            borderColor: 'rgba(127, 140, 141, 0.3)',
        },
        noCommentsIcon: {
            fontSize: 48,
            marginBottom: 15,
        },
        noCommentsText: {
            color: '#7f8c8d',
            fontSize: 16,
            textAlign: 'center',
        },
        commentsList: {
            marginTop: 10,
        },
        commentCard: {
            backgroundColor: '#ffffff',
            borderRadius: 16,
            padding: 20,
            marginBottom: 15,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 4 },
            shadowOpacity: 0.08,
            shadowRadius: 10,
            elevation: 5,
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.1)',
        },

        titleContainer: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 30,
        },
        favoriteButton: {
            width: 50,
            height: 50,
            borderRadius: 25,
            justifyContent: 'center',
            alignItems: 'center',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 3,
        },
        favoriteIcon: {
            fontSize: 24,
            fontWeight: 'bold',
        },
        // Değiştirilmesi gereken style'lar:

        recommendationsSection: {
            marginBottom: 20, // 30'dan 20'ye küçültüldü
            backgroundColor: 'rgba(102, 126, 234, 0.02)',
            borderRadius: 16, // 20'den 16'ya küçültüldü
            padding: 12, // 15'ten 12'ye küçültüldü
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.08)',
        },

        recommendationGrid: {
            flexDirection: 'column',
            maxHeight: 150, // 200'den 150'ye küçültüldü
        },

        recommendationHeader: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 10, // 15'ten 10'a küçültüldü
        },


        recommendationCard: {
            width: (width - 68) / 2, // 74'ten 68'e küçültüldü (daha da sıkışık)
            zIndex: 1,
            elevation: 1,
        },

        recommendationGradient: {
            backgroundColor: '#ffffff',
            borderRadius: 8, // 10'dan 8'e küçültüldü
            padding: 6, // 8'den 6'ya küçültüldü
            shadowColor: '#667eea',
            shadowOffset: { width: 0, height: 1 }, // height 2'den 1'e küçültüldü
            shadowOpacity: 0.06, // 0.08'den 0.06'ya küçültüldü
            shadowRadius: 4, // 6'dan 4'e küçültüldü
            elevation: 3, // 4'ten 3'e küçültüldü
            borderWidth: 1,
            borderColor: 'rgba(102, 126, 234, 0.1)',
            position: 'relative',
            overflow: 'hidden',
            height: 65, // 85'ten 65'e küçültüldü (çok daha küçük)
        },

        recommendationTitle: {
            fontSize: 11, // 12'den 11'e küçültüldü
            fontWeight: '700',
            color: '#2c3e50',
            marginBottom: 2, // 3'ten 2'ye küçültüldü
            lineHeight: 14, // 16'dan 14'e küçültüldü
        },

        recommendationAuthor: {
            fontSize: 9, // 10'dan 9'a küçültüldü
            color: '#7f8c8d',
            marginBottom: 4, // 6'dan 4'e küçültüldü
            fontWeight: '500',
        },

        similarityBar: {
            height: 2, // 3'ten 2'ye küçültüldü
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderRadius: 1, // 2'den 1'e küçültüldü
            overflow: 'hidden',
            marginTop: 'auto',
        },

        recommendationIcon: {
            position: 'absolute',
            top: 4, // 6'dan 4'e küçültüldü
            right: 4, // 6'dan 4'e küçültüldü
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            borderRadius: 6, // 8'den 6'ya küçültüldü
            width: 12, // 16'dan 12'ye küçültüldü
            height: 12, // 16'dan 12'ye küçültüldü
            justifyContent: 'center',
            alignItems: 'center',
        },

        iconText: {
            fontSize: 8, // 9'dan 8'e küçültüldü
        },
        badgeText: {
            color: '#ffffff',
            fontSize: 12,
            fontWeight: '700',
        },
        recommendationScrollContainer: {
            paddingLeft: 5,
            paddingRight: 15,
        },
        recommendationBadge: {
            backgroundColor: '#667eea',
            borderRadius: 15,
            paddingHorizontal: 12,
            paddingVertical: 6,
            shadowColor: '#667eea',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.3,
            shadowRadius: 4,
            elevation: 4,
        },


        recommendationContent: {
            flex: 1,
        },

        similarityContainer: {
            marginTop: 'auto',
        },



        similarityFill: {
            height: '100%',
            backgroundColor: '#667eea',
            borderRadius: 3,
        },


    });

    export default BookDetailsScreen;