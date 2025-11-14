import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    getUsersByHotelStart,
    getUsersByHotelSuccess,
    getUsersByHotelFailure,
    updateUserStart,
    updateUserSuccess,
    updateUserFailure,
    deleteUserStart,
    deleteUserSuccess,
    deleteUserFailure,
} from '../slices/usersSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== USERS APIs ====================

export const fetchUsersByHotel = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getUsersByHotelStart());
        const response = await axios.get(`${API_URL}/users/hotel/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(getUsersByHotelSuccess(response.data.users || response.data));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch users';
        dispatch(getUsersByHotelFailure(message));
        throw error;
    }
};

export const updateUserData = (userId: string, userData: any) => async (dispatch: Dispatch) => {
    try {
        dispatch(updateUserStart());
        const response = await axios.put(`${API_URL}/users/${userId}`, userData, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(updateUserSuccess(response.data.user || response.data));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to update user';
        dispatch(updateUserFailure(message));
        throw error;
    }
};

export const deleteUserData = (userId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(deleteUserStart());
        const response = await axios.delete(`${API_URL}/users/${userId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(deleteUserSuccess(userId));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to delete user';
        dispatch(deleteUserFailure(message));
        throw error;
    }
};

