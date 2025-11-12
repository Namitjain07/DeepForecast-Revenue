import axios from 'axios';
import type {Dispatch} from '@reduxjs/toolkit';
import { loginStart, loginSuccess, loginFailure } from '../slices/authSlice';
import {
    getDashboardStatsStart,
    getDashboardStatsSuccess,
    getDashboardStatsFailure,
} from '../slices/dashboardSlice';
import {
    getAllHotelsStart,
    getAllHotelsSuccess,
    getAllHotelsFailure,
    appendHotels,
    getRecentHotelsStart,
    getRecentHotelsSuccess,
    getRecentHotelsFailure,
    searchHotelsStart,
    searchHotelsSuccess,
    searchHotelsFailure,
} from '../slices/hotelSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== AUTH APIs ====================

export const loginUser = (email: string, password: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(loginStart());
        const response = await axios.post(`${API_URL}/auth/login`, {
            email,
            password,
        });
        dispatch(loginSuccess(response.data));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'An error occurred';
        dispatch(loginFailure(message));
        throw error;
    }
};

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

// ==================== HOTEL APIs ====================

export const fetchAllHotels = (page: number = 1, limit: number = 10) => async (dispatch: Dispatch) => {
    try {
        dispatch(getAllHotelsStart());
        const response = await axios.get(
            `${API_URL}/hotels?page=${page}&limit=${limit}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );

        if (page === 1) {
            dispatch(getAllHotelsSuccess({
                hotels: response.data.hotels,
                pagination: response.data.pagination,
            }));
        } else {
            dispatch(appendHotels({
                hotels: response.data.hotels,
                pagination: response.data.pagination,
            }));
        }
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch hotels';
        dispatch(getAllHotelsFailure(message));
        console.error('Fetch hotels error:', message);
        throw error;
    }
};

export const fetchRecentlyAddedHotels = (limit: number = 3) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRecentHotelsStart());
        const response = await axios.post(
            `${API_URL}/hotels/recently-added`,
            { limit },
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        dispatch(getRecentHotelsSuccess(response.data.hotels));
        return response.data.hotels;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch recent hotels';
        dispatch(getRecentHotelsFailure(message));
        console.error('Fetch recent hotels error:', message);
        throw error;
    }
};

export const searchHotels = (searchTerm: string, page: number = 1, limit: number = 10) => async (dispatch: Dispatch) => {
    try {
        dispatch(searchHotelsStart());
        const response = await axios.post(
            `${API_URL}/hotels/search`,
            { searchTerm, page, limit },
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        dispatch(searchHotelsSuccess({
            hotels: response.data.hotels,
            pagination: response.data.pagination,
        }));

        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to search hotels';
        dispatch(searchHotelsFailure(message));
        console.error('Search hotels error:', message);
        throw error;
    }
};

export const addHotel = (hotelData: any) => async () => {
    try {
        const response = await axios.post(
            `${API_URL}/hotels/add_hotel`,
            hotelData,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to add hotel';
        console.error('Add hotel error:', message);
        throw error;
    }
};



