import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    addLastTrainStart,
    addLastTrainSuccess,
    addLastTrainFailure,
    getLastTrainStart,
    getLastTrainSuccess,
    getLastTrainFailure,
} from '../slices/modelTrainSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== MODEL TRAIN APIs ====================

/**
 * Add a new last train record
 */
export const addLastTrainRecord = (userId: string, hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(addLastTrainStart());
        const response = await axios.post(
            `${API_URL}/train/start`,
            {
                userId,
                hotelId,
            },
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );

        dispatch(addLastTrainSuccess(response.data.lastTrain));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to add train record';
        dispatch(addLastTrainFailure(message));
        console.error('Add last train error:', message);
        throw error;
    }
};

/**
 * Get last train record by hotel ID
 */
export const getLastTrainByHotel = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getLastTrainStart());
        const response = await axios.get(
            `${API_URL}/train/${hotelId}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );

        dispatch(getLastTrainSuccess(response.data.lastTrain));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch train record';
        dispatch(getLastTrainFailure(message));
        console.error('Get last train error:', message);
        throw error;
    }
};

