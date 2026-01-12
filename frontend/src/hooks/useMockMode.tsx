/**
 * Mock Mode Context Provider and Hook
 *
 * Manages application-wide Mock Mode state with localStorage persistence.
 * When Mock Mode is enabled, the API client sends X-Mock-Mode header
 * and the backend returns synthetic demo data.
 *
 * IMPORTANT: When mock mode is toggled, all React Query caches are invalidated
 * to ensure fresh data is fetched from the correct source.
 */
/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from 'react';
import { useQueryClient } from '@tanstack/react-query';

const STORAGE_KEY = 'stella_mock_mode';

interface MockModeContextValue {
    mockMode: boolean;
    setMockMode: (enabled: boolean) => void;
    toggleMockMode: () => void;
}

const MockModeContext = createContext<MockModeContextValue | undefined>(undefined);

interface MockModeProviderProps {
    children: ReactNode;
}

export function MockModeProvider({ children }: MockModeProviderProps) {
    const queryClient = useQueryClient();
    const isFirstRender = useRef(true);

    const [mockMode, setMockModeState] = useState<boolean>(() => {
        // Initialize from localStorage
        if (typeof window !== 'undefined') {
            const stored = localStorage.getItem(STORAGE_KEY);
            return stored === 'true';
        }
        return false;
    });

    // Persist to localStorage and invalidate queries when state changes
    useEffect(() => {
        localStorage.setItem(STORAGE_KEY, mockMode.toString());

        // Skip invalidation on first render (initial load)
        if (isFirstRender.current) {
            isFirstRender.current = false;
            return;
        }

        // Invalidate all queries to force refetch with new data source
        // This ensures mock data is cleared when switching to live mode
        // and live data is cleared when switching to mock mode
        queryClient.invalidateQueries();
    }, [mockMode, queryClient]);

    const setMockMode = useCallback((enabled: boolean) => {
        setMockModeState(enabled);
    }, []);

    const toggleMockMode = useCallback(() => {
        setMockModeState(prev => !prev);
    }, []);

    return (
        <MockModeContext.Provider value={{ mockMode, setMockMode, toggleMockMode }}>
            {children}
        </MockModeContext.Provider>
    );
}

export function useMockMode(): MockModeContextValue {
    const context = useContext(MockModeContext);
    if (context === undefined) {
        throw new Error('useMockMode must be used within a MockModeProvider');
    }
    return context;
}

/**
 * Helper function to get mock mode state from localStorage.
 * Used by the API client which may not have access to React context.
 */
export function getMockModeFromStorage(): boolean {
    if (typeof window !== 'undefined') {
        return localStorage.getItem(STORAGE_KEY) === 'true';
    }
    return false;
}
