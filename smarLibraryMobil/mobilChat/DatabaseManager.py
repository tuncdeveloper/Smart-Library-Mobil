import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        load_dotenv()
        self.connection = None
        self.connect()

    def connect(self):
        """PostgreSQL veritabanına bağlan"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST"),
                database=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                port=int(os.getenv("POSTGRES_PORT", 5432))
            )
            logger.info("✅ Veritabanı bağlantısı başarılı!")
        except Exception as e:
            logger.error(f"❌ Veritabanı bağlantı hatası: {e}")

    def get_books_dataframe(self):
        """Kitap verilerini DataFrame olarak getir"""
        query = """
        SELECT 
            id, title, author, publisher, publication_year, 
            category, language, page_count, description, 
            average_rating, total_ratings
        FROM books
        """

        try:
            df = pd.read_sql_query(query, self.connection)
            return df
        except Exception as e:
            logger.error(f"❌ Veri çekme hatası: {e}")
            return pd.DataFrame()

    def close_connection(self):
        """Veritabanı bağlantısını kapat"""
        if self.connection:
            self.connection.close()


# DatabaseManager sınıfına eklenecek metodlar

def get_books_by_category(self, category, limit=10):
    """Kategoriye göre kitapları getir"""
    try:
        query = """
        SELECT id, title, author, category, average_rating, total_ratings, 
               publication_year, page_count, publisher, description
        FROM books 
        WHERE category = %s 
        ORDER BY average_rating DESC, total_ratings DESC
        LIMIT %s
        """

        self.cursor.execute(query, (category, limit))
        columns = [desc[0] for desc in self.cursor.description]
        results = self.cursor.fetchall()

        return [dict(zip(columns, row)) for row in results]

    except Exception as e:
        logger.error(f"❌ Kategoriye göre kitap getirme hatası: {e}")
        return []


def get_books_by_author(self, author, limit=10):
    """Yazara göre kitapları getir"""
    try:
        query = """
        SELECT id, title, author, category, average_rating, total_ratings, 
               publication_year, page_count, publisher, description
        FROM books 
        WHERE author = %s 
        ORDER BY average_rating DESC, total_ratings DESC
        LIMIT %s
        """

        self.cursor.execute(query, (author, limit))
        columns = [desc[0] for desc in self.cursor.description]
        results = self.cursor.fetchall()

        return [dict(zip(columns, row)) for row in results]

    except Exception as e:
        logger.error(f"❌ Yazara göre kitap getirme hatası: {e}")
        return []


def get_book_by_id(self, book_id):
    """ID'ye göre kitap getir"""
    try:
        query = """
        SELECT id, title, author, category, average_rating, total_ratings, 
               publication_year, page_count, publisher, description
        FROM books 
        WHERE id = %s
        """

        self.cursor.execute(query, (book_id,))
        result = self.cursor.fetchone()

        if result:
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, result))
        return None

    except Exception as e:
        logger.error(f"❌ ID'ye göre kitap getirme hatası: {e}")
        return None


def get_book_by_title(self, title):
    """Başlığa göre kitap getir (fuzzy match)"""
    try:
        # Önce tam eşleşme dene
        query = """
        SELECT id, title, author, category, average_rating, total_ratings, 
               publication_year, page_count, publisher, description
        FROM books 
        WHERE LOWER(title) = LOWER(%s)
        LIMIT 1
        """

        self.cursor.execute(query, (title,))
        result = self.cursor.fetchone()

        if result:
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, result))

        # Tam eşleşme yoksa benzer başlık ara
        query = """
        SELECT id, title, author, category, average_rating, total_ratings, 
               publication_year, page_count, publisher, description
        FROM books 
        WHERE LOWER(title) LIKE LOWER(%s)
        ORDER BY LENGTH(title)
        LIMIT 1
        """

        self.cursor.execute(query, (f'%{title}%',))
        result = self.cursor.fetchone()

        if result:
            columns = [desc[0] for desc in self.cursor.description]
            return dict(zip(columns, result))

        return None

    except Exception as e:
        logger.error(f"❌ Başlığa göre kitap getirme hatası: {e}")
        return None