import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    getRecentRecordsStart,
    getRecentRecordsSuccess,
    getRecentRecordsFailure,
    getDateRangeRecordsStart,
    getDateRangeRecordsSuccess,
    getDateRangeRecordsFailure,
} from '../slices/recordsSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== RECORDS APIs ====================

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

