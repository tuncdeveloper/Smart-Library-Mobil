import React, { useEffect, useRef } from 'react';
import { Animated, Text, TouchableOpacity, View } from 'react-native';
import { BookListDTO } from '../types/book';

type Props = {
    item: BookListDTO;
    index: number;
    onPress: () => void;
    isFavorite: boolean;
    onToggleFavorite: () => void;
};

const BookItem: React.FC<Props> = ({ item, index, onPress, isFavorite, onToggleFavorite }) => {
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const slideAnim = useRef(new Animated.Value(30)).current;

    useEffect(() => {
        Animated.parallel([
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 500,
                delay: index * 100,
                useNativeDriver: true,
            }),
            Animated.timing(slideAnim, {
                toValue: 0,
                duration: 500,
                delay: index * 100,
                useNativeDriver: true,
            }),
        ]).start();
    }, []);

    return (
        <Animated.View
            style={{
                opacity: fadeAnim,
                transform: [{ translateY: slideAnim }],
                margin: 8,
                borderRadius: 10,
                overflow: 'hidden',
            }}
        >
            <TouchableOpacity onPress={onPress} style={{ padding: 16, backgroundColor: '#fff' }}>
                <Text>{item.title}</Text>
                <Text>{isFavorite ? '❤️' : '🤍'}</Text>
                <TouchableOpacity onPress={onToggleFavorite}>
                    <Text>Favori</Text>
                </TouchableOpacity>
            </TouchableOpacity>
        </Animated.View>
    );
};

export default BookItem;
