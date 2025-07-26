export type BookCreateDTO = {
    title: string;
    author: string;
    publisher: string;
    publicationYear: number;
    category: string;
    language: string;
    pageCount: number;

    description: string;
};

export type BookDetailDTO = {
    id: number;

    title: string;
    author: string;
    publisher: string;
    publicationYear: number;
    category: string;
    language: string;
    pageCount: number;

    description: string;
    addedDate: string; // ISO date
    averageRating: number;
};



export type BookListDTO = {
    id: number;
    title: string;
    author: string;
    category: string;
    language: string;

    averageRating?: number; // <- Bu satırı ekle

};

export type BookFavoriteDTO = {
    studentId: number;
    bookId: number;
    title: string;
    author: string;
    category: string;
    publicationYear: number;
};


export type BookUpdateDTO = {
    title: string;
    author: string;
    publisher: string;
    publicationYear: number;
    category: string;
    language: string;
    pageCount: number;

    location: string;
    description: string;

};

// Yeni eklenen rating DTO'ları
export type BookRatingRequestDTO = {
    bookId: number;
    studentId: number;
    comment?: string;      // Kullanıcının yazdığı yorum
    rating: number;
};

export type BookRatingResponseDTO = {
    id: number;           // Veritabanındaki puanlama ID'si
    bookId: number;
    studentId: number;
    studentUsername: string;    // kullnıcı adı
    bookTitle: string; // kitap başlığı
    rating: number;
    comment?: string;      // Kullanıcının yazdığı yorum
    createdAt: string;    // ISO date string olarak
};

export type BookRatingDetailDTO = {
    bookId: number;
    bookTitle: string;
    averageRating: number | null;
    totalRatings: number;
    starCount: number;
};

export interface ModelStatsDTO {
    totalBooks: number;
    uniqueAuthors: number;
    uniqueCategories: number;
    modelFeatures: number;
    hyperparameters: { [key: string]: any };
    confidenceThreshold: number;
}

// Recommendation Types
export interface BookRecommendationDTO {
    rank: number;
    title: string;
    author: string;
    category: string;
    publisher: string;
    publicationYear: number;
    averageRating: number;
    totalRatings: number;
    pageCount: number;
    description: string;
    similarityScore: number;
    confidenceScore: number;
    diversityScore: number;
    finalScore: number;
    confidenceDetails: { [key: string]: number };
    recommendationReasons: string[];
}

export interface TargetBookDTO {
    title: string;
    author: string;
    category: string;
    publisher: string;
    publicationYear: number;
    averageRating: number;
    totalRatings: number;
    description: string;
}
//book.ts
export interface RecommendationResponseDTO {
    success: boolean;
    message: string;
    targetBook?: TargetBookDTO;
    recommendations?: BookRecommendationDTO[];
    totalRecommendations: number;
}

// Add to existing ModelStatsDTO
