import React, { ReactNode, useEffect, useState } from 'react';
import { View, Text, StyleSheet, Platform, StatusBar, Animated, Dimensions, TouchableOpacity } from 'react-native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import { LinearGradient } from 'expo-linear-gradient';
import { useAuth } from '@/src/contexts/AuthContext';
import LoginPage from '../screens/LoginScreen';
import RegisterPage from '../screens/RegisterScreen';
import StudentScreen from '../screens/StudentScreen';
import BookListScreen from '../screens/BookListScreen';
import BookFavoriteScreen from '../screens/BookFavoriteScreen';
import StudentInformation from '../screens/StudentInformation';
import BookDetail from '../screens/BookDetailsScreen';

const Stack = createNativeStackNavigator();
const Tab = createMaterialTopTabNavigator();
const { width, height } = Dimensions.get('window');

type GradientBackgroundProps = {
    children: ReactNode;
};

function GradientBackground({ children }: GradientBackgroundProps) {
    const [animatedValue] = useState(new Animated.Value(0));

    useEffect(() => {
        const animate = () => {
            Animated.sequence([
                Animated.timing(animatedValue, {
                    toValue: 1,
                    duration: 3000,
                    useNativeDriver: false,
                }),
                Animated.timing(animatedValue, {
                    toValue: 0,
                    duration: 3000,
                    useNativeDriver: false,
                }),
            ]).start(() => animate());
        };
        animate();
    }, []);

    const interpolatedColor = animatedValue.interpolate({
        inputRange: [0, 1],
        outputRange: ['#667eea', '#764ba2'],
    });

    return (
        <View style={styles.gradientBackground}>
            <LinearGradient
                colors={['#667eea', '#764ba2', '#f093fb']}
                style={styles.backgroundGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            />
            {/* Animated Background Elements */}
            <Animated.View style={[styles.floatingCircle1, { backgroundColor: interpolatedColor }]} />
            <Animated.View style={[styles.floatingCircle2, { backgroundColor: interpolatedColor }]} />
            <Animated.View style={[styles.floatingCircle3, { backgroundColor: interpolatedColor }]} />
            {children}
        </View>
    );
}

type TabLabelProps = {
    label: string;
    focused: boolean;
    icon: string;
};

function TabLabel({ label, focused, icon }: TabLabelProps) {
    const [scaleAnim] = useState(new Animated.Value(1));

    useEffect(() => {
        if (focused) {
            Animated.sequence([
                Animated.timing(scaleAnim, {
                    toValue: 1.1,
                    duration: 150,
                    useNativeDriver: true,
                }),
                Animated.timing(scaleAnim, {
                    toValue: 1,
                    duration: 150,
                    useNativeDriver: true,
                }),
            ]).start();
        }
    }, [focused]);

    return (
        <Animated.View style={[styles.tabLabelContainer, { transform: [{ scale: scaleAnim }] }]}>
            <Text style={[styles.tabIcon, focused && styles.tabIconActive]}>{icon}</Text>
            <Text style={[styles.tabLabel, focused && styles.tabLabelActive]}>
                {label}
            </Text>
            {focused && <View style={styles.tabActiveDot} />}
        </Animated.View>
    );
}

function HomeStack() {
    return (
        <Stack.Navigator
            screenOptions={{
                headerShown: false,
                contentStyle: styles.sceneContainer
            }}
        >
            <Stack.Screen name="HomeScreen" component={StudentScreen} />
            <Stack.Screen name="BookDetail" component={BookDetail} />
        </Stack.Navigator>
    );
}

function BooksStack() {
    return (
        <Stack.Navigator
            screenOptions={{
                headerShown: false,
                contentStyle: styles.sceneContainer
            }}
        >
            <Stack.Screen name="BookListScreen" component={BookListScreen} />
            <Stack.Screen name="BookDetail" component={BookDetail} />
        </Stack.Navigator>
    );
}

function FavoritesStack() {
    return (
        <Stack.Navigator
            screenOptions={{
                headerShown: false,
                contentStyle: styles.sceneContainer
            }}
        >
            <Stack.Screen name="BookFavoriteScreen" component={BookFavoriteScreen} />
            <Stack.Screen name="BookDetail" component={BookDetail} />
        </Stack.Navigator>
    );
}

function ProfileStack() {
    return (
        <Stack.Navigator
            screenOptions={{
                headerShown: false,
                contentStyle: styles.sceneContainer
            }}
        >
            <Stack.Screen name="StudentInformation" component={StudentInformation} />
        </Stack.Navigator>
    );
}

function CustomTabBar({ state, descriptors, navigation }: any) {
    const [fadeAnim] = useState(new Animated.Value(0));

    useEffect(() => {
        Animated.timing(fadeAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
        }).start();
    }, []);

    const tabs = [
        { key: 'Home', label: 'Ana Sayfa', icon: '🏠' },
        { key: 'Books', label: 'Kitaplar', icon: '📚' },
        { key: 'Favorites', label: 'Favoriler', icon: '❤️' },
        { key: 'Profile', label: 'Profil', icon: '👤' },
    ];

    return (
        <Animated.View style={[styles.customTabBar, { opacity: fadeAnim }]}>
            <LinearGradient
                colors={['rgba(102, 126, 234, 0.9)', 'rgba(118, 75, 162, 0.8)', 'rgba(240, 147, 251, 0.7)']}
                style={styles.tabBarGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
            >
                <View style={styles.tabBarInner}>
                    {tabs.map((tab, index) => {
                        const isFocused = state.index === index;
                        const onPress = () => {
                            const event = navigation.emit({
                                type: 'tabPress',
                                target: state.routes[index].key,
                                canPreventDefault: true,
                            });

                            if (!isFocused && !event.defaultPrevented) {
                                navigation.navigate(tab.key);
                            }
                        };

                        return (
                            <TouchableOpacity
                                key={tab.key}
                                onPress={onPress}
                                style={[styles.tabItem, isFocused && styles.tabItemActive]}
                                activeOpacity={0.7}
                            >
                                <View style={[styles.tabItemInner, isFocused && styles.tabItemInnerActive]}>
                                    <Text style={[styles.tabItemIcon, isFocused && styles.tabItemIconActive]}>
                                        {tab.icon}
                                    </Text>
                                    <Text style={[styles.tabItemLabel, isFocused && styles.tabItemLabelActive]}>
                                        {tab.label}
                                    </Text>
                                </View>
                                {isFocused && <View style={styles.tabItemIndicator} />}
                            </TouchableOpacity>
                        );
                    })}
                </View>
            </LinearGradient>
        </Animated.View>
    );
}

function MainTabs() {
    return (
        <GradientBackground>
            <Tab.Navigator
                tabBar={(props) => <CustomTabBar {...props} />}
                screenOptions={{
                    tabBarStyle: { display: 'none' },
                }}
                tabBarPosition="top"
            >
                <Tab.Screen
                    name="Home"
                    component={HomeStack}
                    options={{
                        tabBarLabel: ({ focused }: { focused: boolean }) =>
                            <TabLabel label="Ana Sayfa" focused={focused} icon="🏠" />,
                    }}
                />
                <Tab.Screen
                    name="Books"
                    component={BooksStack}
                    options={{
                        tabBarLabel: ({ focused }: { focused: boolean }) =>
                            <TabLabel label="Kitaplar" focused={focused} icon="📚" />,
                    }}
                />
                <Tab.Screen
                    name="Favorites"
                    component={FavoritesStack}
                    options={{
                        tabBarLabel: ({ focused }: { focused: boolean }) =>
                            <TabLabel label="Favoriler" focused={focused} icon="❤️" />,
                    }}
                />
                <Tab.Screen
                    name="Profile"
                    component={ProfileStack}
                    options={{
                        tabBarLabel: ({ focused }: { focused: boolean }) =>
                            <TabLabel label="Profil" focused={focused} icon="👤" />,
                    }}
                />
            </Tab.Navigator>
        </GradientBackground>
    );
}

function AuthStack() {
    const [slideAnim] = useState(new Animated.Value(100));

    useEffect(() => {
        Animated.timing(slideAnim, {
            toValue: 0,
            duration: 800,
            useNativeDriver: true,
        }).start();
    }, []);

    return (
        <GradientBackground>
            <Animated.View style={[styles.authContainer, { transform: [{ translateY: slideAnim }] }]}>
                <Stack.Navigator screenOptions={{ headerShown: false }}>
                    <Stack.Screen name="Login" component={LoginPage} />
                    <Stack.Screen name="Register" component={RegisterPage} />
                </Stack.Navigator>
            </Animated.View>
        </GradientBackground>
    );
}

function LoadingScreen() {
    const [rotateAnim] = useState(new Animated.Value(0));
    const [pulseAnim] = useState(new Animated.Value(1));

    useEffect(() => {
        const rotateAnimation = Animated.loop(
            Animated.timing(rotateAnim, {
                toValue: 1,
                duration: 2000,
                useNativeDriver: true,
            })
        );

        const pulseAnimation = Animated.loop(
            Animated.sequence([
                Animated.timing(pulseAnim, {
                    toValue: 1.2,
                    duration: 1000,
                    useNativeDriver: true,
                }),
                Animated.timing(pulseAnim, {
                    toValue: 1,
                    duration: 1000,
                    useNativeDriver: true,
                }),
            ])
        );

        rotateAnimation.start();
        pulseAnimation.start();

        return () => {
            rotateAnimation.stop();
            pulseAnimation.stop();
        };
    }, []);

    const rotate = rotateAnim.interpolate({
        inputRange: [0, 1],
        outputRange: ['0deg', '360deg'],
    });

    return (
        <GradientBackground>
            <View style={styles.loadingContainer}>
                <Animated.View
                    style={[
                        styles.loadingSpinner,
                        {
                            transform: [
                                { rotate },
                                { scale: pulseAnim }
                            ]
                        }
                    ]}
                >
                    <Text style={styles.loadingIcon}>📚</Text>
                </Animated.View>
                <Text style={styles.loadingText}>Yükleniyor...</Text>
                <View style={styles.loadingDots}>
                    <View style={[styles.loadingDot, styles.loadingDot1]} />
                    <View style={[styles.loadingDot, styles.loadingDot2]} />
                    <View style={[styles.loadingDot, styles.loadingDot3]} />
                </View>
            </View>
        </GradientBackground>
    );
}

export default function AppRoutes() {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return <LoadingScreen />;
    }

    return (
        <View style={styles.container}>
            <StatusBar
                barStyle="light-content"
                backgroundColor="transparent"
                translucent
            />
            <Stack.Navigator screenOptions={{ headerShown: false }}>
                {!isAuthenticated ? (
                    <Stack.Screen name="AuthStack" component={AuthStack} />
                ) : (
                    <Stack.Screen name="MainTabs" component={MainTabs} />
                )}
            </Stack.Navigator>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: 'transparent',
    },
    gradientBackground: {
        flex: 1,
        position: 'relative',
    },
    backgroundGradient: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
    },
    floatingCircle1: {
        position: 'absolute',
        top: -100,
        right: -100,
        width: 200,
        height: 200,
        borderRadius: 100,
        opacity: 0.1,
    },
    floatingCircle2: {
        position: 'absolute',
        bottom: -80,
        left: -80,
        width: 160,
        height: 160,
        borderRadius: 80,
        opacity: 0.08,
    },
    floatingCircle3: {
        position: 'absolute',
        top: height * 0.4,
        left: -50,
        width: 100,
        height: 100,
        borderRadius: 50,
        opacity: 0.12,
    },
    authContainer: {
        flex: 1,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    loadingSpinner: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: 'rgba(255, 255, 255, 0.2)',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
    },
    loadingIcon: {
        fontSize: 32,
    },
    loadingText: {
        fontSize: 18,
        color: '#ffffff',
        fontWeight: '600',
        marginBottom: 20,
    },
    loadingDots: {
        flexDirection: 'row',
        gap: 8,
    },
    loadingDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: 'rgba(255, 255, 255, 0.6)',
    },
    loadingDot1: {
        opacity: 0.4,
    },
    loadingDot2: {
        opacity: 0.7,
    },
    loadingDot3: {
        opacity: 1,
    },
    customTabBar: {
        position: 'absolute',
        top: -36,
        left: 0,
        right: 0,
        paddingHorizontal: 16,
        paddingTop: Platform.OS === 'ios' ? 50 : (StatusBar.currentHeight || 0) + 10,
        paddingBottom: 12,
        zIndex: 1000,
    },
    tabBarGradient: {
        borderRadius: 24,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.3,
        shadowRadius: 20,
        elevation: 15,
    },
    tabBarInner: {
        flexDirection: 'row',
        paddingVertical: 12,
        paddingHorizontal: 8,
        backgroundColor: 'rgba(255, 255, 255, 0.15)',
        backdropFilter: 'blur(10px)',
    },
    tabItem: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 8,
        paddingHorizontal: 4,
        borderRadius: 16,
        position: 'relative',
    },
    tabItemActive: {
        backgroundColor: 'rgba(255, 255, 255, 0.25)',
        shadowColor: 'rgba(255, 255, 255, 0.3)',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.8,
        shadowRadius: 8,
        elevation: 5,
    },
    tabItemInner: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    tabItemInnerActive: {
        transform: [{ scale: 1.05 }],
    },
    tabItemIcon: {
        fontSize: 24,
        marginBottom: 3,
        opacity: 0.7,
    },
    tabItemIconActive: {
        opacity: 1,
        textShadowColor: 'rgba(255, 255, 255, 0.3)',
        textShadowOffset: { width: 0, height: 1 },
        textShadowRadius: 2,
    },
    tabItemLabel: {
        fontSize: 13,
        fontWeight: '600',
        color: 'rgba(255, 255, 255, 0.8)',
        textAlign: 'center',
    },
    tabItemLabelActive: {
        color: '#ffffff',
        fontWeight: '700',
        textShadowColor: 'rgba(255, 255, 255, 0.3)',
        textShadowOffset: { width: 0, height: 1 },
        textShadowRadius: 2,
    },
    tabItemIndicator: {
        position: 'absolute',
        bottom: -8,
        width: 32,
        height: 4,
        backgroundColor: '#ffffff',
        borderRadius: 2,
        shadowColor: '#ffffff',
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.8,
        shadowRadius: 6,
        elevation: 6,
    },
    sceneContainer: {
        flex: 1,
        paddingTop: Platform.OS === 'ios' ? 110 : 100,
    },
    // Legacy styles for compatibility
    tabBar: {
        display: 'none',
    },
    tabBarLabel: {
        display: 'none',
    },
    tabBarIndicator: {
        display: 'none',
    },
    tabBarItem: {
        display: 'none',
    },
    tabLabelContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 4,
    },
    tabIcon: {
        fontSize: 18,
        marginBottom: 2,
        opacity: 0.7,
    },
    tabIconActive: {
        opacity: 1,
    },
    tabLabel: {
        fontSize: 11,
        fontWeight: '600',
        color: 'rgba(255, 255, 255, 0.7)',
        textAlign: 'center',
    },
    tabLabelActive: {
        color: '#ffffff',
        fontWeight: '700',
    },
    tabActiveDot: {
        width: 4,
        height: 4,
        borderRadius: 2,
        backgroundColor: '#ffffff',
        marginTop: 2,
    },
});