import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface DashboardStats {
    totalHotels: number;
    totalUsers: number;
    recentHotels: number;
    recentUsers: number;
    lastUpdated: string;
}

interface DashboardState {
    stats: DashboardStats | null;
    loading: boolean;
    error: string | null;
}

const initialState: DashboardState = {
    stats: null,
    loading: false,
    error: null,
};

const dashboardSlice = createSlice({
    name: 'dashboard',
    initialState,
    reducers: {
        getDashboardStatsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDashboardStatsSuccess: (state, action: PayloadAction<DashboardStats>) => {
            state.loading = false;
            state.stats = action.payload;
        },
        getDashboardStatsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },
        clearError: (state) => {
            state.error = null;
        },
    },
});

export const {
    getDashboardStatsStart,
    getDashboardStatsSuccess,
    getDashboardStatsFailure,
    clearError,
} = dashboardSlice.actions;

export default dashboardSlice.reducer;

