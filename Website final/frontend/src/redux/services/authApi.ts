import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import { loginStart, loginSuccess, loginFailure, addNewUserStart, addNewUserSuccess, addNewUserFailure } from '../slices/authSlice';

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

/**
 * Add a new user (owner or manager)
 */
export const addNewUser = (userData: {
    name: string;
    email: string;
    password: string;
    hotelId: string;
    role: 'owner' | 'manager';
}) => async (dispatch: Dispatch) => {
    try {
        dispatch(addNewUserStart());
        const response = await axios.post(`${API_URL}/auth/add_user`, userData, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        dispatch(addNewUserSuccess());
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to add user';
        dispatch(addNewUserFailure(message));
        console.error('Add user error:', message);
        throw error;
    }
};
