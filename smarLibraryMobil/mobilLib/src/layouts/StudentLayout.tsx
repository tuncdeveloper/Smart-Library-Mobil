// StudentLayout.js
import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import ProfileScreen from '../screens/StudentScreen';
import BooksScreen from '../screens/BookListScreen';
import FavoritesScreen from '../screens/BookFavoriteScreen';
import UpdateInfoScreen from '../screens/StudentInformation';
import BookDetail from '../screens/BookDetailsScreen';

const Stack = createNativeStackNavigator();

// Menü öğeleri
const menuItems = [
    { key: 'profile', label: 'Ana Sayfa' },
    { key: 'books', label: 'Kitaplar' },
    { key: 'favorites', label: 'Favorilerim' },
    { key: 'update', label: 'Bilgileri Güncelle' },
];

// Her menü için ayrı Stack Navigator
const ProfileStack = () => (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="ProfileMain" component={ProfileScreen} />
    </Stack.Navigator>
);

const BooksStack = () => (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="BooksMain" component={BooksScreen} />
        <Stack.Screen name="BookDetail" component={BookDetail} />
    </Stack.Navigator>
);

const FavoritesStack = () => (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="FavoritesMain" component={FavoritesScreen} />
        <Stack.Screen name="BookDetail" component={BookDetail} />
    </Stack.Navigator>
);

const UpdateStack = () => (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="UpdateMain" component={UpdateInfoScreen} />
    </Stack.Navigator>
);

const StudentLayout = () => {
    const [activeMenu, setActiveMenu] = useState('profile');

    // Aktif menüye göre hangi stack gösterilecek
    const renderContent = () => {
        switch (activeMenu) {
            case 'profile':
                return <ProfileStack />;
            case 'books':
                return <BooksStack />;
            case 'favorites':
                return <FavoritesStack />;
            case 'update':
                return <UpdateStack />;
            default:
                return <ProfileStack />;
        }
    };

    return (
        <View style={styles.container}>
            {/* Üst Menü Bar - Her zaman görünür */}
            <View style={styles.header}>
                {menuItems.map((item) => (
                    <TouchableOpacity
                        key={item.key}
                        onPress={() => setActiveMenu(item.key)}
                        style={[
                            styles.menuItem,
                            activeMenu === item.key && styles.menuItemActive,
                        ]}
                    >
                        <Text
                            style={[
                                styles.menuText,
                                activeMenu === item.key && styles.menuTextActive,
                            ]}
                        >
                            {item.label}
                        </Text>
                    </TouchableOpacity>
                ))}
            </View>

            {/* Alt içerik */}
            <View style={styles.content}>{renderContent()}</View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1 },

    header: {
        height: 60,
        backgroundColor: '#3b82f6',
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        elevation: 4,
        shadowColor: '#000',
        shadowOpacity: 0.2,
        shadowOffset: { width: 0, height: 2 },
        zIndex: 1000, // Üst menünün her zaman en üstte kalması için
    },

    menuItem: {
        paddingVertical: 10,
        paddingHorizontal: 15,
        borderRadius: 6,
    },

    menuItemActive: {
        backgroundColor: '#1dfa15',
    },

    menuText: {
        color: 'white',
        fontWeight: '600',
    },

    menuTextActive: {
        color: '#1f2937',
    },

    content: {
        flex: 1,
        backgroundColor: '#f0f4f8',
    },
});

export default StudentLayout;