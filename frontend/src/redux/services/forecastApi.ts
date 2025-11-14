import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    getAvailableDatesStart,
    getAvailableDatesSuccess,
    getAvailableDatesFailure,
    getDateRangeForecastsStart,
    getDateRangeForecastsSuccess,
    getDateRangeForecastsFailure,
} from '../slices/forcastSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== FORECAST APIs ====================

export const fetchForecastAvailableDates = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getAvailableDatesStart());
        const response = await axios.get(`${API_URL}/forecast/available-dates/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });

        const dates = response.data.dates || [];
        dispatch(getAvailableDatesSuccess({
            dates,
            minDate: response.data.minDate,
            maxDate: response.data.maxDate,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch available forecast dates';
        dispatch(getAvailableDatesFailure(message));
        throw error;
    }
};

export const downloadForecastCSV = (
    hotelId: string,
    startDate: string,
    endDate: string
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDateRangeForecastsStart());

        let allForecasts: any[] = [];
        let page = 1;
        let hasMorePages = true;
        const limit = 100;

        // Fetch all forecasts by paginating
        while (hasMorePages) {
            const response = await axios.post(
                `${API_URL}/forecast/date-range`,
                {
                    hotelId,
                    startDate,
                    endDate,
                    page,
                    limit,
                },
                {
                    headers: {
                        'Authorization': `Bearer ${getToken()}`,
                        'Content-Type': 'application/json',
                    },
                }
            );

            const forecasts = response.data.forecasts || [];
            allForecasts = [...allForecasts, ...forecasts];

            const pagination = response.data.pagination || {};
            hasMorePages = pagination.hasNextPage || false;
            page++;
        }

        if (allForecasts.length === 0) {
            throw new Error('No forecast data found for the selected date range');
        }

        const headers = [
            'Date',
            'Revenue',
            'Rooms Sold',
            'Arrival Rooms',
            'Departure Rooms',
            'OOO Rooms'
        ];

        const csvContent = [
            headers.join(','),
            ...allForecasts.map((forecast: any) =>
                [
                    new Date(forecast.date).toLocaleDateString('en-US'),
                    forecast.revenue,
                    forecast.roomSold,
                    forecast.arrivalRoom,
                    forecast.departureRoom,
                    forecast.oooRoom
                ].join(',')
            )
        ].join('\n');

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', `forecast_${startDate}_to_${endDate}.csv`);
        document.body.appendChild(link);
        link.click();
        link.parentNode?.removeChild(link);

        dispatch(getDateRangeForecastsSuccess({
            forecasts: allForecasts,
            count: allForecasts.length,
        }));
        return { forecasts: allForecasts, count: allForecasts.length };
    } catch (error: any) {
        const message = error.response?.data?.message || error.message || 'Failed to download CSV';
        dispatch(getDateRangeForecastsFailure(message));
        throw error;
    }
};

