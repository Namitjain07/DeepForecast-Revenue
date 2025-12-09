import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    // Revenue Metrics
    getRevenueStart,
    getRevenueSuccess,
    getRevenueFailure,
    // Room Sold Metrics
    getRoomSoldStart,
    getRoomSoldSuccess,
    getRoomSoldFailure,
    // Arrival Metrics
    getArrivalStart,
    getArrivalSuccess,
    getArrivalFailure,
    // Departure Metrics
    getDepartureStart,
    getDepartureSuccess,
    getDepartureFailure,
    // OOO Metrics
    getOOOStart,
    getOOOSuccess,
    getOOOFailure,
} from '../slices/recordsSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== METRIC-SPECIFIC RECORDS APIs ====================

// Revenue APIs
export const fetchRevenueData = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRevenueStart());
        const response = await axios.get(
            `${API_URL}/records/revenue/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.data || [];
        dispatch(getRevenueSuccess({
            records,
            count: records.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch revenue data';
        dispatch(getRevenueFailure(message));
        throw error;
    }
};

// Room Sold APIs
export const fetchRoomSoldData = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRoomSoldStart());
        const response = await axios.get(
            `${API_URL}/records/room-sold/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.data || [];
        dispatch(getRoomSoldSuccess({
            records,
            count: records.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch room sold data';
        dispatch(getRoomSoldFailure(message));
        throw error;
    }
};

// Arrival APIs
export const fetchArrivalData = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getArrivalStart());
        const response = await axios.get(
            `${API_URL}/records/arrival/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.data || [];
        dispatch(getArrivalSuccess({
            records,
            count: records.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch arrival data';
        dispatch(getArrivalFailure(message));
        throw error;
    }
};

// Departure APIs
export const fetchDepartureData = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDepartureStart());
        const response = await axios.get(
            `${API_URL}/records/departure/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.data || [];
        dispatch(getDepartureSuccess({
            records,
            count: records.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch departure data';
        dispatch(getDepartureFailure(message));
        throw error;
    }
};

// OOO APIs
export const fetchOOOData = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getOOOStart());
        const response = await axios.get(
            `${API_URL}/records/ooo/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.data || [];
        dispatch(getOOOSuccess({
            records,
            count: records.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch OOO data';
        dispatch(getOOOFailure(message));
        throw error;
    }
};

