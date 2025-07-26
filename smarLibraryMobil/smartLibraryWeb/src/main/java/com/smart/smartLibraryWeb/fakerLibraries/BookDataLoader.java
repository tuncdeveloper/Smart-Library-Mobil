//package com.smart.smartLibraryWeb.fakerLibraries;
//
//import com.smart.smartLibraryWeb.model.Book;
//import com.smart.smartLibraryWeb.repository.BookRepository;
//import jakarta.annotation.PostConstruct;
//import lombok.RequiredArgsConstructor;
//import org.springframework.stereotype.Component;
//
//import java.time.LocalDateTime;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
//@Component
//@RequiredArgsConstructor
//public class BookDataLoader {
//
//    private final BookRepository bookRepository;
//
//    private final String[] categories = {
//            "Macera", "Aşk / Romantik", "Dram", "Tarihi Roman", "Polisiye / Suç",
//            "Bilim Kurgu (Sci-Fi)", "Fantastik", "Gerilim / Thriller", "Psikolojik"
//    };
//
//
//    private static class BookData {
//        String title;
//        String author;
//        String category;
//        String publisher;
//        Integer publicationYear;
//        String language;
//        Integer pageCount;
//
//        public BookData(String title, String author, String category, String publisher, Integer publicationYear, String language, Integer pageCount) {
//            this.title = title;
//            this.author = author;
//            this.category = category;
//            this.publisher = publisher;
//            this.publicationYear = publicationYear;
//            this.language = language;
//            this.pageCount = pageCount;
//        }
//    }
////    private final List<BookData> bookDataList = List.of(
////            new BookData("Kırmızı Saçlı Kadın", "Orhan Pamuk", "Psikolojik", "Yapı Kredi Yayınları", 2016, "Türkçe", 204),
////            new BookData("Masumiyet Müzesi", "Orhan Pamuk", "Aşk / Romantik", "İletişim Yayınları", 2008, "Türkçe", 592),
////            new BookData("Beyaz Kale", "Orhan Pamuk", "Tarihi Roman", "İletişim Yayınları", 1985, "Türkçe", 160),
////            new BookData("İnce Memed", "Yaşar Kemal", "Macera", "Yapı Kredi Yayınları", 1955, "Türkçe", 430),
////            new BookData("Yer Demir Gök Bakır", "Yaşar Kemal", "Toplumcu Gerçekçi", "Yapı Kredi Yayınları", 1963, "Türkçe", 320),
////            new BookData("Tutunamayanlar", "Oğuz Atay", "Psikolojik", "İletişim Yayınları", 1972, "Türkçe", 724),
////            new BookData("Tehlikeli Oyunlar", "Oğuz Atay", "Psikolojik", "İletişim Yayınları", 1973, "Türkçe", 471),
////            new BookData("Suç ve Ceza", "Fyodor Dostoyevski", "Psikolojik", "Can Yayınları", 1866, "Türkçe", 687),
////            new BookData("Karamazov Kardeşler", "Fyodor Dostoyevski", "Felsefi Roman", "Can Yayınları", 1880, "Türkçe", 840),
////            new BookData("1984", "George Orwell", "Bilim Kurgu (Sci-Fi)", "Can Yayınları", 1949, "Türkçe", 352),
////            new BookData("Hayvan Çiftliği", "George Orwell", "Siyasi Alegori", "Can Yayınları", 1945, "Türkçe", 152),
////            new BookData("Simyacı", "Paulo Coelho", "Fantastik", "Can Yayınları", 1988, "Türkçe", 188),
////            new BookData("Veronika Ölmek İstiyor", "Paulo Coelho", "Psikolojik", "Can Yayınları", 1998, "Türkçe", 224),
////            new BookData("Savaş ve Barış", "Lev Tolstoy", "Tarihi Roman", "İletişim Yayınları", 1869, "Türkçe", 1296),
////            new BookData("Anna Karenina", "Lev Tolstoy", "Aşk / Romantik", "Can Yayınları", 1877, "Türkçe", 864),
////            new BookData("Yabancı", "Albert Camus", "Varoluşçu", "Can Yayınları", 1942, "Türkçe", 160),
////            new BookData("Veba", "Albert Camus", "Felsefi Roman", "Can Yayınları", 1947, "Türkçe", 320),
////            new BookData("Kürk Mantolu Madonna", "Sabahattin Ali", "Aşk / Romantik", "Yapı Kredi Yayınları", 1943, "Türkçe", 160),
////            new BookData("İçimizdeki Şeytan", "Sabahattin Ali", "Psikolojik", "Yapı Kredi Yayınları", 1940, "Türkçe", 200),
////            new BookData("Saatleri Ayarlama Enstitüsü", "Ahmet Hamdi Tanpınar", "Modern", "Dergah Yayınları", 1961, "Türkçe", 400),
////            new BookData("Huzur", "Ahmet Hamdi Tanpınar", "Psikolojik", "Dergah Yayınları", 1949, "Türkçe", 350),
////            new BookData("Aşk", "Elif Şafak", "Aşk / Tasavvufi", "Doğan Kitap", 2009, "Türkçe", 420),
////            new BookData("Baba ve Piç", "Elif Şafak", "Toplumsal Roman", "Doğan Kitap", 2006, "Türkçe", 368),
////            new BookData("Uçurtma Avcısı", "Khaled Hosseini", "Dram", "Everest Yayınları", 2003, "Türkçe", 384),
////            new BookData("Bin Muhteşem Güneş", "Khaled Hosseini", "Dram", "Everest Yayınları", 2007, "Türkçe", 432),
////            new BookData("Yüzüklerin Efendisi: Yüzük Kardeşliği", "J.R.R. Tolkien", "Fantastik", "Metis Yayınları", 1954, "Türkçe", 523),
////            new BookData("Harry Potter ve Felsefe Taşı", "J.K. Rowling", "Fantastik", "Yapı Kredi Yayınları", 1997, "Türkçe", 223),
////            new BookData("Grinin Elli Tonu", "E. L. James", "Aşk / Erotik", "Pegasus Yayınları", 2011, "Türkçe", 528)
////    );
//
//    private final List<BookData> bookDataList = List.of(
//            // Önceki 28 kitap burada olacak
//            // Yeni eklenen 72 kitap:
//
//            new BookData("Dönüşüm", "Franz Kafka", "Psikolojik", "Can Yayınları", 1915, "Türkçe", 104),
//            new BookData("Milena'ya Mektuplar", "Franz Kafka", "Mektup", "Can Yayınları", 1952, "Türkçe", 240),
//            new BookData("Şato", "Franz Kafka", "Modern Klasik", "Can Yayınları", 1926, "Türkçe", 344),
//            new BookData("Bülbülü Öldürmek", "Harper Lee", "Dram", "Sel Yayıncılık", 1960, "Türkçe", 357),
//            new BookData("Fareler ve İnsanlar", "John Steinbeck", "Dram", "Sel Yayıncılık", 1937, "Türkçe", 111),
//            new BookData("Gazap Üzümleri", "John Steinbeck", "Toplumcu Gerçekçi", "Sel Yayıncılık", 1939, "Türkçe", 464),
//            new BookData("Martı", "Jonathan Livingston", "Felsefi", "Epsilon Yayınları", 1970, "Türkçe", 96),
//            new BookData("Küçük Prens", "Antoine de Saint-Exupéry", "Çocuk Klasik", "Can Çocuk Yayınları", 1943, "Türkçe", 112),
//            new BookData("Cesur Yeni Dünya", "Aldous Huxley", "Bilim Kurgu", "İthaki Yayınları", 1932, "Türkçe", 288),
//            new BookData("Fahrenheit 451", "Ray Bradbury", "Bilim Kurgu", "İthaki Yayınları", 1953, "Türkçe", 208),
//            new BookData("Dava", "Franz Kafka", "Modern Klasik", "Can Yayınları", 1925, "Türkçe", 224),
//            new BookData("Madam Bovary", "Gustave Flaubert", "Realist Roman", "İş Bankası Kültür Yayınları", 1856, "Türkçe", 368),
//            new BookData("Sefiller", "Victor Hugo", "Tarihi Roman", "İş Bankası Kültür Yayınları", 1862, "Türkçe", 1463),
//            new BookData("Notre Dame'ın Kamburu", "Victor Hugo", "Tarihi Roman", "İş Bankası Kültür Yayınları", 1831, "Türkçe", 528),
//            new BookData("Genç Werther'in Acıları", "Johann Wolfgang von Goethe", "Mektup Roman", "İş Bankası Kültür Yayınları", 1774, "Türkçe", 144),
//            new BookData("Don Kişot", "Miguel de Cervantes", "Klasik", "İş Bankası Kültür Yayınları", 1605, "Türkçe", 976),
//            new BookData("Yüzyıllık Yalnızlık", "Gabriel García Márquez", "Büyülü Gerçekçilik", "Can Yayınları", 1967, "Türkçe", 464),
//            new BookData("Körlük", "José Saramago", "Distopik", "Kırmızı Kedi Yayınları", 1995, "Türkçe", 344),
//            new BookData("Kumarbaz", "Fyodor Dostoyevski", "Psikolojik", "İletişim Yayınları", 1866, "Türkçe", 224),
//            new BookData("Budala", "Fyodor Dostoyevski", "Psikolojik", "İletişim Yayınları", 1869, "Türkçe", 857),
//            new BookData("Ecinniler", "Fyodor Dostoyevski", "Siyasi Roman", "İletişim Yayınları", 1872, "Türkçe", 816),
//            new BookData("Ölü Canlar", "Nikolay Gogol", "Sosyal Eleştiri", "İletişim Yayınları", 1842, "Türkçe", 448),
//            new BookData("Martin Eden", "Jack London", "Otobiyografik Roman", "İş Bankası Kültür Yayınları", 1909, "Türkçe", 488),
//            new BookData("Beyaz Diş", "Jack London", "Macera", "İş Bankası Kültür Yayınları", 1906, "Türkçe", 256),
//            new BookData("İki Şehrin Hikayesi", "Charles Dickens", "Tarihi Roman", "Can Yayınları", 1859, "Türkçe", 464),
//            new BookData("Büyük Umutlar", "Charles Dickens", "Klasik", "Can Yayınları", 1861, "Türkçe", 544),
//            new BookData("Moby Dick", "Herman Melville", "Macera", "İş Bankası Kültür Yayınları", 1851, "Türkçe", 720),
//            new BookData("Gurur ve Önyargı", "Jane Austen", "Aşk / Romantik", "Can Yayınları", 1813, "Türkçe", 352),
//            new BookData("Uğultulu Tepeler", "Emily Brontë", "Gotik Roman", "İş Bankası Kültür Yayınları", 1847, "Türkçe", 384),
//            new BookData("Jane Eyre", "Charlotte Brontë", "Bildungsroman", "İş Bankası Kültür Yayınları", 1847, "Türkçe", 532),
//            new BookData("Siddhartha", "Hermann Hesse", "Felsefi Roman", "Can Yayınları", 1922, "Türkçe", 152),
//            new BookData("Bozkırkurdu", "Hermann Hesse", "Psikolojik", "Can Yayınları", 1927, "Türkçe", 224),
//            new BookData("Denemeler", "Michel de Montaigne", "Deneme", "İş Bankası Kültür Yayınları", 1580, "Türkçe", 480),
//            new BookData("Yeraltından Notlar", "Fyodor Dostoyevski", "Psikolojik", "İletişim Yayınları", 1864, "Türkçe", 176),
//            new BookData("Koku", "Patrick Süskind", "Tarihi Roman", "Can Yayınları", 1985, "Türkçe", 264),
//            new BookData("Olağanüstü Bir Gece", "Stefan Zweig", "Novella", "Can Yayınları", 1922, "Türkçe", 80),
//            new BookData("Satranç", "Stefan Zweig", "Novella", "Can Yayınları", 1942, "Türkçe", 72),
//            new BookData("Amok Koşucusu", "Stefan Zweig", "Novella", "Can Yayınları", 1922, "Türkçe", 64),
//            new BookData("Dokuzuncu Hariciye Koğuşu", "Peyami Safa", "Psikolojik", "Ötüken Neşriyat", 1930, "Türkçe", 160),
//            new BookData("Fatih-Harbiye", "Peyami Safa", "Toplumsal Roman", "Ötüken Neşriyat", 1931, "Türkçe", 160),
//            new BookData("Yalnızız", "Peyami Safa", "Psikolojik", "Ötüken Neşriyat", 1951, "Türkçe", 352),
//            new BookData("Aganta Burina Burinata", "Halikarnas Balıkçısı", "Deneme", "Bilgi Yayınevi", 1946, "Türkçe", 184),
//            new BookData("Ölmez Otu", "Yaşar Kemal", "Roman", "Yapı Kredi Yayınları", 1968, "Türkçe", 384),
//            new BookData("Demirciler Çarşısı Cinayeti", "Yaşar Kemal", "Polisiye", "Yapı Kredi Yayınları", 1974, "Türkçe", 336),
//            new BookData("Bir Ada Hikayesi", "Yaşar Kemal", "Roman", "Yapı Kredi Yayınları", 1997, "Türkçe", 432),
//            new BookData("Aylak Adam", "Yusuf Atılgan", "Modern Roman", "Yapı Kredi Yayınları", 1959, "Türkçe", 144),
//            new BookData("Anayurt Oteli", "Yusuf Atılgan", "Psikolojik", "Yapı Kredi Yayınları", 1973, "Türkçe", 128),
//            new BookData("Korkuyu Beklerken", "Oğuz Atay", "Öykü", "İletişim Yayınları", 1975, "Türkçe", 144),
//            new BookData("Eylül", "Mehmet Rauf", "Aşk / Romantik", "İletişim Yayınları", 1901, "Türkçe", 240),
//            new BookData("Araba Sevdası", "Recaizade Mahmut Ekrem", "Realist Roman", "İletişim Yayınları", 1896, "Türkçe", 248),
//            new BookData("Sergüzeşt", "Samipaşazade Sezai", "Romantik Roman", "İletişim Yayınları", 1889, "Türkçe", 160),
//            new BookData("Felatun Bey ile Rakım Efendi", "Ahmet Mithat Efendi", "Roman", "İletişim Yayınları", 1875, "Türkçe", 256),
//            new BookData("Parasız Yatılı", "Füruzan", "Öykü", "Yapı Kredi Yayınları", 1971, "Türkçe", 160),
//            new BookData("Benim Adım Kırmızı", "Orhan Pamuk", "Tarihi Roman", "Yapı Kredi Yayınları", 1998, "Türkçe", 504),
//            new BookData("Kar", "Orhan Pamuk", "Siyasi Roman", "İletişim Yayınları", 2002, "Türkçe", 440),
//            new BookData("Yeni Hayat", "Orhan Pamuk", "Roman", "İletişim Yayınları", 1994, "Türkçe", 304),
//            new BookData("Kara Kitap", "Orhan Pamuk", "Roman", "İletişim Yayınları", 1990, "Türkçe", 464),
//            new BookData("Cevdet Bey ve Oğulları", "Orhan Pamuk", "Aile Romanı", "İletişim Yayınları", 1982, "Türkçe", 576),
//            new BookData("Sessiz Ev", "Orhan Pamuk", "Roman", "İletişim Yayınları", 1983, "Türkçe", 360),
//            new BookData("Serenad", "Zülfü Livaneli", "Tarihi Roman", "Remzi Kitabevi", 2011, "Türkçe", 368),
//            new BookData("Huzursuzluk", "Zülfü Livaneli", "Roman", "Doğan Kitap", 2017, "Türkçe", 304),
//            new BookData("Leyla'nın Evi", "Zülfü Livaneli", "Roman", "Doğan Kitap", 2011, "Türkçe", 336),
//            new BookData("Son Ada", "Zülfü Livaneli", "Distopik", "Doğan Kitap", 2008, "Türkçe", 280),
//            new BookData("Kardeşimin Hikayesi", "Zülfü Livaneli", "Roman", "Doğan Kitap", 2013, "Türkçe", 288),
//            new BookData("Bir Kedi, Bir Adam, Bir Ölüm", "Zülfü Livaneli", "Roman", "Doğan Kitap", 2001, "Türkçe", 200),
//            new BookData("Engereğin Gözündeki Kamaşma", "Zülfü Livaneli", "Roman", "Doğan Kitap", 1976, "Türkçe", 224),
//            new BookData("Kardeşimin Hikayesi", "Zülfü Livaneli", "Roman", "Doğan Kitap", 2013, "Türkçe", 288),
//            new BookData("Körlük", "José Saramago", "Distopik", "Kırmızı Kedi Yayınları", 1995, "Türkçe", 344),
//            new BookData("Görmek", "José Saramago", "Distopik", "Kırmızı Kedi Yayınları", 2004, "Türkçe", 352),
//            new BookData("Ricardo Reis'in Öldüğü Yıl", "José Saramago", "Roman", "Kırmızı Kedi Yayınları", 1984, "Türkçe", 320),
//            new BookData("Bütün İsimler", "José Saramago", "Roman", "Kırmızı Kedi Yayınları", 1997, "Türkçe", 240),
//            new BookData("Mağara", "José Saramago", "Roman", "Kırmızı Kedi Yayınları", 2000, "Türkçe", 256),
//            new BookData("Ölüm Bir Varmış Bir Yokmuş", "José Saramago", "Roman", "Kırmızı Kedi Yayınları", 2005, "Türkçe", 208),
//            new BookData("Kabil", "José Saramago", "Roman", "Kırmızı Kedi Yayınları", 2009, "Türkçe", 192)
//
//    );
//
//    @PostConstruct
//    public void loadBooks() {
//        if (bookRepository.count() > 0) return;
//
//        List<Book> books = new ArrayList<>();
//        Random random = new Random();
//
//        for (BookData data : bookDataList) {
//            Book book = new Book();
//            book.setTitle(data.title);
//            book.setAuthor(data.author);
//            book.setCategory(data.category);
//            book.setPublisher(data.publisher);
//            book.setPublicationYear(data.publicationYear);
//            book.setLanguage(data.language);
//            book.setPageCount(data.pageCount);
//            book.setDescription(data.title + " kitabı, " + data.author + " tarafından yazılmış etkileyici bir " + data.category.toLowerCase() + " romanıdır.");
//            book.setAddedDate(LocalDateTime.now());
//            books.add(book);
//        }
//
//
//        bookRepository.saveAll(books);
//        System.out.println(books.size() + " kitap başarıyla yüklendi.");
//    }
//}
