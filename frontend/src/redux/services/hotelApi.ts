import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
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
    getGeneralInfoStart,
    getGeneralInfoSuccess,
    getGeneralInfoFailure,
    addHotelStart,
    addHotelSuccess,
    addHotelFailure,
    getDashboardStatsStart,
    getDashboardStatsSuccess,
    getDashboardStatsFailure,
    updateHotelInfoStart,
    updateHotelInfoSuccess,
    updateHotelInfoFailure,
} from '../slices/hotelSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

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

export const addHotel = (hotelData: any) => async (dispatch: Dispatch) => {
    try {
        dispatch(addHotelStart());
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
        dispatch(addHotelSuccess(response.data.hotel || response.data));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to add hotel';
        dispatch(addHotelFailure(message));
        console.error('Add hotel error:', message);
        throw error;
    }
};

// ==================== HOTEL DETAILS APIs ====================

export const fetchGeneralInfo = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getGeneralInfoStart());
        const response = await axios.get(`${API_URL}/hotels/general-info/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(getGeneralInfoSuccess(response.data.hotel));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch general info';
        dispatch(getGeneralInfoFailure(message));
        throw error;
    }
};

/**
 * Fetch hotel dashboard stats (Total Revenue, Rooms Sold, Avg Occupancy for last 30 days)
 */
export const fetchHotelDashboardStats = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDashboardStatsStart());
        const response = await axios.get(`${API_URL}/hotels/dashboard-stats/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(getDashboardStatsSuccess());
        return response.data.stats;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch hotel dashboard stats';
        dispatch(getDashboardStatsFailure(message));
        console.error('Hotel dashboard stats error:', message);
        throw error;
    }
};

/**
 * Update hotel information
 */
export const updateHotelInfo = (hotelId: string, hotelData: any) => async (dispatch: Dispatch) => {
    try {
        dispatch(updateHotelInfoStart());
        const response = await axios.put(
            `${API_URL}/hotels/update/${hotelId}`,
            hotelData,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        dispatch(updateHotelInfoSuccess(response.data.hotel));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to update hotel info';
        dispatch(updateHotelInfoFailure(message));
        console.error('Update hotel info error:', message);
        throw error;
    }
};



