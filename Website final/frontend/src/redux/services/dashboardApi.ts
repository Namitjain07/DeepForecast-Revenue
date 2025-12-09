import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    getDashboardStatsStart,
    getDashboardStatsSuccess,
    getDashboardStatsFailure,
} from '../slices/dashboardSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== DASHBOARD APIs ====================

export const fetchDashboardStats = () => async (dispatch: Dispatch) => {
    try {
        dispatch(getDashboardStatsStart());
        const response = await axios.get(`${API_URL}/admin/dashboard/stats`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(getDashboardStatsSuccess(response.data.stats));
        return response.data.stats;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch dashboard stats';
        dispatch(getDashboardStatsFailure(message));
        console.error('Dashboard stats error:', message);
        throw error;
    }
};

