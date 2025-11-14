import { createSlice} from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface MetricRecord {
    date: string;
    value: number;
}

export interface Record {
    id: string;
    date: string;
    roomsSold: number;
    day: string;
    arrivalRooms: number;
    departureRooms: number;
    oooRooms: number;
    occupancyPercentage: number;
    roomRevenue: number;
    averageRoomRate: number;
    complimentRooms: number;
    houseUse: number;
    individualConfirm: number;
    pax: number;
    totalRoomInventory: number;
}

interface RecordsState {
    recentRecords: Record[];
    dateRangeRecords: Record[];
    // Revenue
    revenue: MetricRecord[];
    // Room Sold
    roomSold: MetricRecord[];
    // Arrival
    arrival: MetricRecord[];
    // Departure
    departure: MetricRecord[];
    // OOO
    ooo: MetricRecord[];
    loading: boolean;
    error: string | null;
    count: number;
}

const initialState: RecordsState = {
    recentRecords: [],
    dateRangeRecords: [],
    revenue: [],
    roomSold: [],
    arrival: [],
    departure: [],
    ooo: [],
    loading: false,
    error: null,
    count: 0,
};

const recordsSlice = createSlice({
    name: 'records',
    initialState,
    reducers: {
        // Get Recent Records
        getRecentRecordsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRecentRecordsSuccess: (state, action: PayloadAction<{ records: Record[]; count: number }>) => {
            state.loading = false;
            state.recentRecords = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getRecentRecordsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Get Records by Date Range
        getDateRangeRecordsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDateRangeRecordsSuccess: (state, action: PayloadAction<{ records: Record[]; count: number }>) => {
            state.loading = false;
            state.dateRangeRecords = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getDateRangeRecordsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Revenue Metrics
        getRevenueStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRevenueSuccess: (state, action: PayloadAction<{ records: MetricRecord[]; count: number }>) => {
            state.loading = false;
            state.revenue = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getRevenueFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Room Sold Metrics
        getRoomSoldStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRoomSoldSuccess: (state, action: PayloadAction<{ records: MetricRecord[]; count: number }>) => {
            state.loading = false;
            state.roomSold = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getRoomSoldFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Arrival Metrics
        getArrivalStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getArrivalSuccess: (state, action: PayloadAction<{ records: MetricRecord[]; count: number }>) => {
            state.loading = false;
            state.arrival = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getArrivalFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Departure Metrics
        getDepartureStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDepartureSuccess: (state, action: PayloadAction<{ records: MetricRecord[]; count: number }>) => {
            state.loading = false;
            state.departure = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getDepartureFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // OOO Metrics
        getOOOStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getOOOSuccess: (state, action: PayloadAction<{ records: MetricRecord[]; count: number }>) => {
            state.loading = false;
            state.ooo = action.payload.records;
            state.count = action.payload.count;
            state.error = null;
        },
        getOOOFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        resetRecords: (state) => {
            state.recentRecords = [];
            state.dateRangeRecords = [];
            state.revenue = [];
            state.roomSold = [];
            state.arrival = [];
            state.departure = [];
            state.ooo = [];
            state.loading = false;
            state.error = null;
            state.count = 0;
        },
    },
});

export const {
    getRecentRecordsStart,
    getRecentRecordsSuccess,
    getRecentRecordsFailure,
    getDateRangeRecordsStart,
    getDateRangeRecordsSuccess,
    getDateRangeRecordsFailure,
    // Revenue
    getRevenueStart,
    getRevenueSuccess,
    getRevenueFailure,
    // Room Sold
    getRoomSoldStart,
    getRoomSoldSuccess,
    getRoomSoldFailure,
    // Arrival
    getArrivalStart,
    getArrivalSuccess,
    getArrivalFailure,
    // Departure
    getDepartureStart,
    getDepartureSuccess,
    getDepartureFailure,
    // OOO
    getOOOStart,
    getOOOSuccess,
    getOOOFailure,
    resetRecords,
} = recordsSlice.actions;

export default recordsSlice.reducer;
