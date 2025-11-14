import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

interface AuthState {
    token: string | null;
    user: {
        _id?: string;
        name?: string;
        email?: string;
        role?: string;
    } | null;
    isAuthenticated: boolean;
    loading: boolean;
    error: string | null;
}

// Helper function to get persisted auth state
const getPersistedAuthState = () => {
    try {
        const token = localStorage.getItem('token');
        const user = localStorage.getItem('user');

        if (token && user) {
            return {
                token,
                user: JSON.parse(user),
                isAuthenticated: true,
            };
        }
    } catch (error) {
        console.error('Error retrieving persisted auth state:', error);
    }

    return {
        token: null,
        user: null,
        isAuthenticated: false,
    };
};

const persistedState = getPersistedAuthState();

const initialState: AuthState = {
    token: persistedState.token,
    user: persistedState.user,
    isAuthenticated: persistedState.isAuthenticated,
    loading: false,
    error: null,
};

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        loginStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        loginSuccess: (state, action: PayloadAction<{ token: string; user: any }>) => {
            state.loading = false;
            state.isAuthenticated = true;
            state.token = action.payload.token;
            state.user = action.payload.user;

            // Persist to localStorage
            localStorage.setItem('token', action.payload.token);
            localStorage.setItem('user', JSON.stringify(action.payload.user));
        },
        loginFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
            state.isAuthenticated = false;
            state.token = null;
            state.user = null;

            // Clear localStorage
            localStorage.removeItem('token');
            localStorage.removeItem('user');
        },
        logout: (state) => {
            state.token = null;
            state.user = null;
            state.isAuthenticated = false;

            // Clear localStorage
            localStorage.removeItem('token');
            localStorage.removeItem('user');
        },
        // Restore auth state from localStorage on app initialization
        restoreAuthState: (state) => {
            const persistedState = getPersistedAuthState();
            state.token = persistedState.token;
            state.user = persistedState.user;
            state.isAuthenticated = persistedState.isAuthenticated;
        },
    },
});

export const { loginStart, loginSuccess, loginFailure, logout, restoreAuthState } = authSlice.actions;
export default authSlice.reducer;
