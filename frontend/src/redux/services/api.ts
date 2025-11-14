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
    getGeneralInfoStart,
    getGeneralInfoSuccess,
    getGeneralInfoFailure,
} from '../slices/hotelSlice';
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
import {
    getRecentRecordsStart,
    getRecentRecordsSuccess,
    getRecentRecordsFailure,
    getDateRangeRecordsStart,
    getDateRangeRecordsSuccess,
    getDateRangeRecordsFailure,
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
import {
    getAvailableDatesStart,
    getAvailableDatesSuccess,
    getAvailableDatesFailure,
    getDateRangeForecastsStart,
    getDateRangeForecastsSuccess,
    getDateRangeForecastsFailure,
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




// ==================== SIMPLIFIED METRIC-SPECIFIC APIs ====================

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





export const fetchAvailableDates = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDateRangeRecordsStart());
        const response = await axios.get(`${API_URL}/records/available-dates/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });

        const dates = response.data.dates || [];
        dispatch(getDateRangeRecordsSuccess({
            records: [],
            count: dates.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch available dates';
        dispatch(getDateRangeRecordsFailure(message));
        throw error;
    }
};

export const fetchRecentRecords = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(getRecentRecordsStart());
        const response = await axios.get(`${API_URL}/records/recent/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });
        const records = response.data.records || response.data;
        dispatch(getRecentRecordsSuccess({
            records,
            count: Array.isArray(records) ? records.length : 0,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch recent records';
        dispatch(getRecentRecordsFailure(message));
        throw error;
    }
};

export const fetchRecordsByDateRange = (
    hotelId: string,
    startDate: string,
    endDate: string
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDateRangeRecordsStart());
        const response = await axios.post(
            `${API_URL}/records/date-range`,
            {
                hotelId,
                startDate,
                endDate,
            },
            {
                headers: {
                    'Authorization': `Bearer ${getToken()}`,
                    'Content-Type': 'application/json',
                },
            }
        );
        const records = response.data.records || response.data;
        dispatch(getDateRangeRecordsSuccess({
            records,
            count: Array.isArray(records) ? records.length : 0,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch records';
        dispatch(getDateRangeRecordsFailure(message));
        throw error;
    }
};

export const downloadRecordsCSV = (
    hotelId: string,
    startDate: string,
    endDate: string
) => async (dispatch: Dispatch) => {
    try {
        dispatch(getDateRangeRecordsStart());

        let allRecords: any[] = [];
        let page = 1;
        let hasMorePages = true;
        const limit = 100; // Fetch 100 records per request

        // Fetch all records by paginating
        while (hasMorePages) {
            const response = await axios.post(
                `${API_URL}/records/date-range`,
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

            const records = response.data.records || [];
            allRecords = [...allRecords, ...records];

            // Check if there are more pages
            const pagination = response.data.pagination || {};
            hasMorePages = pagination.hasNextPage || false;
            page++;
        }

        // Create CSV content
        if (allRecords.length === 0) {
            throw new Error('No records found for the selected date range');
        }

        const headers = [
            'Date',
            'Day',
            'Rooms Sold',
            'Arrival Rooms',
            'Departure Rooms',
            'OOO Rooms',
            'Occupancy %',
            'Room Revenue',
            'Avg Room Rate',
            'PAX',
            'Compliment Rooms',
            'House Use',
            'Individual Confirm',
            'Total Room Inventory'
        ];

        const csvContent = [
            headers.join(','),
            ...allRecords.map((record: any) =>
                [
                    new Date(record.date).toLocaleDateString('en-US'),
                    record.day,
                    record.roomsSold,
                    record.arrivalRooms,
                    record.departureRooms,
                    record.oooRooms,
                    record.occupancyPercentage.toFixed(2),
                    record.roomRevenue,
                    record.averageRoomRate.toFixed(2),
                    record.pax,
                    record.complimentRooms,
                    record.houseUse,
                    record.individualConfirm,
                    record.totalRoomInventory
                ].join(',')
            )
        ].join('\n');

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.setAttribute('download', `records_${startDate}_to_${endDate}.csv`);
        document.body.appendChild(link);
        link.click();
        link.parentNode?.removeChild(link);

        dispatch(getDateRangeRecordsSuccess({
            records: allRecords,
            count: allRecords.length,
        }));
        return { records: allRecords, count: allRecords.length };
    } catch (error: any) {
        const message = error.response?.data?.message || error.message || 'Failed to download CSV';
        dispatch(getDateRangeRecordsFailure(message));
        throw error;
    }
};

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

