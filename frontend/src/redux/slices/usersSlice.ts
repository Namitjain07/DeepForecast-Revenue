import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface User {
    id: string;
    name: string;
    email: string;
    role: 'owner' | 'manager';
    hotelId?: string;
}

interface UsersState {
    users: User[];
    loading: boolean;
    error: string | null;
    selectedUser: User | null;
}

const initialState: UsersState = {
    users: [],
    loading: false,
    error: null,
    selectedUser: null,
};

const usersSlice = createSlice({
    name: 'users',
    initialState,
    reducers: {
        // Get Users by Hotel
        getUsersByHotelStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getUsersByHotelSuccess: (state, action: PayloadAction<User[]>) => {
            state.loading = false;
            state.users = action.payload;
            state.error = null;
        },
        getUsersByHotelFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Update User
        updateUserStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        updateUserSuccess: (state, action: PayloadAction<User>) => {
            state.loading = false;
            const index = state.users.findIndex(u => u.id === action.payload.id);
            if (index !== -1) {
                state.users[index] = action.payload;
            }
            state.error = null;
        },
        updateUserFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Delete User
        deleteUserStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        deleteUserSuccess: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.users = state.users.filter(u => u.id !== action.payload);
            state.error = null;
        },
        deleteUserFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        resetUsers: (state) => {
            state.users = [];
            state.loading = false;
            state.error = null;
            state.selectedUser = null;
        },
    },
});

export const {
    getUsersByHotelStart,
    getUsersByHotelSuccess,
    getUsersByHotelFailure,
    updateUserStart,
    updateUserSuccess,
    updateUserFailure,
    deleteUserStart,
    deleteUserSuccess,
    deleteUserFailure,
    resetUsers,
} = usersSlice.actions;

export default usersSlice.reducer;

