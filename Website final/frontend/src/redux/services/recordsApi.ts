import axios from 'axios';
import type { Dispatch } from '@reduxjs/toolkit';
import {
    getRecentRecordsStart,
    getRecentRecordsSuccess,
    getRecentRecordsFailure,
    addRecordsToHotelStart,
    addRecordsToHotelSuccess,
    addRecordsToHotelFailure,
    fetchAvailableDatesStart,
    fetchAvailableDatesSuccess,
    fetchAvailableDatesFailure,
    downloadRecordsCSVStart,
    downloadRecordsCSVSuccess,
    downloadRecordsCSVFailure,
} from '../slices/recordsSlice';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';

const getToken = () => localStorage.getItem('token');

// ==================== RECORDS APIs ====================

/**
 * Add multiple records to a hotel from CSV/XLSX upload
 */
export const addRecordsToHotel = (data: { hotelId: string; records: any[] }) => async (dispatch: Dispatch) => {
    try {
        dispatch(addRecordsToHotelStart());
        const response = await axios.post(`${API_URL}/records/add`, data, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });

        const records = response.data.records || [];
        const count = response.data.count || records.length;

        dispatch(addRecordsToHotelSuccess({
            records: records,
            count: count,
        }));

        return {
            success: true,
            count: count,
            records: records,
            message: response.data.message
        };
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to add records';
        console.log('Error adding records:', error);

        dispatch(addRecordsToHotelFailure(message));
        throw error;
    }
};

export const fetchAvailableDates = (hotelId: string) => async (dispatch: Dispatch) => {
    try {
        dispatch(fetchAvailableDatesStart());
        const response = await axios.get(`${API_URL}/records/available-dates/${hotelId}`, {
            headers: {
                'Authorization': `Bearer ${getToken()}`,
                'Content-Type': 'application/json',
            },
        });

        const dates = response.data.dates || [];
        dispatch(fetchAvailableDatesSuccess({
            records: [],
            count: dates.length,
        }));
        return response.data;
    } catch (error: any) {
        const message = error.response?.data?.message || 'Failed to fetch available dates';
        dispatch(fetchAvailableDatesFailure(message));
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



export const downloadRecordsCSV = (
    hotelId: string,
    startDate: string,
    endDate: string
) => async (dispatch: Dispatch) => {
    try {
        dispatch(downloadRecordsCSVStart());

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
            'ARR',
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

        dispatch(downloadRecordsCSVSuccess({
            records: allRecords,
            count: allRecords.length,
        }));
        return { records: allRecords, count: allRecords.length };
    } catch (error: any) {
        const message = error.response?.data?.message || error.message || 'Failed to download CSV';
        dispatch(downloadRecordsCSVFailure(message));
        throw error;
    }
};
