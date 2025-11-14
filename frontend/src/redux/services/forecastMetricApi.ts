import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    // Revenue Forecast Metrics
    getRevenueForecaastStart,
    getRevenueForecaastSuccess,
    getRevenueForecaastFailure,
    // Room Sold Forecast Metrics
    getRoomSoldForecaastStart,
    getRoomSoldForecaastSuccess,
    getRoomSoldForecaastFailure,
    // Arrival Forecast Metrics
    getArrivalForecaastStart,
    getArrivalForecaastSuccess,
    getArrivalForecaastFailure,
    // Departure Forecast Metrics
    getDepartureForecaastStart,
    getDepartureForecaastSuccess,
    getDepartureForecaastFailure,
    // OOO Forecast Metrics
    getOOOForecaastStart,
    getOOOForecaastSuccess,
    getOOOForecaastFailure,
} from '../slices/forcastSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== METRIC-SPECIFIC FORECAST APIs ====================

// Revenue Forecast APIs
export const fetchRevenueForecasts = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRevenueForecaastStart());
        const response = await axios.get(
            `${API_URL}/forecast/revenue/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const forecasts = response.data.data || [];
        dispatch(getRevenueForecaastSuccess({
            forecasts,
            count: forecasts.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch revenue forecasts';
        dispatch(getRevenueForecaastFailure(message));
        throw error;
    }
};

// Room Sold Forecast APIs
export const fetchRoomSoldForecasts = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRoomSoldForecaastStart());
        const response = await axios.get(
            `${API_URL}/forecast/room-sold/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const forecasts = response.data.data || [];
        dispatch(getRoomSoldForecaastSuccess({
            forecasts,
            count: forecasts.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch room sold forecasts';
        dispatch(getRoomSoldForecaastFailure(message));
        throw error;
    }
};

// Arrival Forecast APIs
export const fetchArrivalForecasts = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getArrivalForecaastStart());
        const response = await axios.get(
            `${API_URL}/forecast/arrival/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const forecasts = response.data.data || [];
        dispatch(getArrivalForecaastSuccess({
            forecasts,
            count: forecasts.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch arrival forecasts';
        dispatch(getArrivalForecaastFailure(message));
        throw error;
    }
};

// Departure Forecast APIs
export const fetchDepartureForecasts = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDepartureForecaastStart());
        const response = await axios.get(
            `${API_URL}/forecast/departure/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const forecasts = response.data.data || [];
        dispatch(getDepartureForecaastSuccess({
            forecasts,
            count: forecasts.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch departure forecasts';
        dispatch(getDepartureForecaastFailure(message));
        throw error;
    }
};

// OOO Forecast APIs
export const fetchOOOForecasts = (
    hotelId: string,
    period: '1w' | '1m' | '3m' | '6m' | '12m'
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getOOOForecaastStart());
        const response = await axios.get(
            `${API_URL}/forecast/ooo/${hotelId}?period=${period}`,
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const forecasts = response.data.data || [];
        dispatch(getOOOForecaastSuccess({
            forecasts,
            count: forecasts.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch OOO forecasts';
        dispatch(getOOOForecaastFailure(message));
        throw error;
    }
};

