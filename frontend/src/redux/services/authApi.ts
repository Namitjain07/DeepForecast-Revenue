import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import { loginStart, loginSuccess, loginFailure } from '../slices/authSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

// const getToken = () => localStorage.getItem('token');

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

