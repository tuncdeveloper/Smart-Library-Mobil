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
//    private final List<BookData> bookDataList = List.of(
//            new BookData("Kırmızı Saçlı Kadın", "Orhan Pamuk", "Psikolojik", "Yapı Kredi Yayınları", 2016, "Türkçe", 204),
//            new BookData("Masumiyet Müzesi", "Orhan Pamuk", "Aşk / Romantik", "İletişim Yayınları", 2008, "Türkçe", 592),
//            new BookData("Beyaz Kale", "Orhan Pamuk", "Tarihi Roman", "İletişim Yayınları", 1985, "Türkçe", 160),
//            new BookData("İnce Memed", "Yaşar Kemal", "Macera", "Yapı Kredi Yayınları", 1955, "Türkçe", 430),
//            new BookData("Yer Demir Gök Bakır", "Yaşar Kemal", "Toplumcu Gerçekçi", "Yapı Kredi Yayınları", 1963, "Türkçe", 320),
//            new BookData("Tutunamayanlar", "Oğuz Atay", "Psikolojik", "İletişim Yayınları", 1972, "Türkçe", 724),
//            new BookData("Tehlikeli Oyunlar", "Oğuz Atay", "Psikolojik", "İletişim Yayınları", 1973, "Türkçe", 471),
//            new BookData("Suç ve Ceza", "Fyodor Dostoyevski", "Psikolojik", "Can Yayınları", 1866, "Türkçe", 687),
//            new BookData("Karamazov Kardeşler", "Fyodor Dostoyevski", "Felsefi Roman", "Can Yayınları", 1880, "Türkçe", 840),
//            new BookData("1984", "George Orwell", "Bilim Kurgu (Sci-Fi)", "Can Yayınları", 1949, "Türkçe", 352),
//            new BookData("Hayvan Çiftliği", "George Orwell", "Siyasi Alegori", "Can Yayınları", 1945, "Türkçe", 152),
//            new BookData("Simyacı", "Paulo Coelho", "Fantastik", "Can Yayınları", 1988, "Türkçe", 188),
//            new BookData("Veronika Ölmek İstiyor", "Paulo Coelho", "Psikolojik", "Can Yayınları", 1998, "Türkçe", 224),
//            new BookData("Savaş ve Barış", "Lev Tolstoy", "Tarihi Roman", "İletişim Yayınları", 1869, "Türkçe", 1296),
//            new BookData("Anna Karenina", "Lev Tolstoy", "Aşk / Romantik", "Can Yayınları", 1877, "Türkçe", 864),
//            new BookData("Yabancı", "Albert Camus", "Varoluşçu", "Can Yayınları", 1942, "Türkçe", 160),
//            new BookData("Veba", "Albert Camus", "Felsefi Roman", "Can Yayınları", 1947, "Türkçe", 320),
//            new BookData("Kürk Mantolu Madonna", "Sabahattin Ali", "Aşk / Romantik", "Yapı Kredi Yayınları", 1943, "Türkçe", 160),
//            new BookData("İçimizdeki Şeytan", "Sabahattin Ali", "Psikolojik", "Yapı Kredi Yayınları", 1940, "Türkçe", 200),
//            new BookData("Saatleri Ayarlama Enstitüsü", "Ahmet Hamdi Tanpınar", "Modern", "Dergah Yayınları", 1961, "Türkçe", 400),
//            new BookData("Huzur", "Ahmet Hamdi Tanpınar", "Psikolojik", "Dergah Yayınları", 1949, "Türkçe", 350),
//            new BookData("Aşk", "Elif Şafak", "Aşk / Tasavvufi", "Doğan Kitap", 2009, "Türkçe", 420),
//            new BookData("Baba ve Piç", "Elif Şafak", "Toplumsal Roman", "Doğan Kitap", 2006, "Türkçe", 368),
//            new BookData("Uçurtma Avcısı", "Khaled Hosseini", "Dram", "Everest Yayınları", 2003, "Türkçe", 384),
//            new BookData("Bin Muhteşem Güneş", "Khaled Hosseini", "Dram", "Everest Yayınları", 2007, "Türkçe", 432),
//            new BookData("Yüzüklerin Efendisi: Yüzük Kardeşliği", "J.R.R. Tolkien", "Fantastik", "Metis Yayınları", 1954, "Türkçe", 523),
//            new BookData("Harry Potter ve Felsefe Taşı", "J.K. Rowling", "Fantastik", "Yapı Kredi Yayınları", 1997, "Türkçe", 223),
//            new BookData("Grinin Elli Tonu", "E. L. James", "Aşk / Erotik", "Pegasus Yayınları", 2011, "Türkçe", 528)
//    );
//
//
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
