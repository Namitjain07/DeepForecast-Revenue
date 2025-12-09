import { createSlice } from '@reduxjs/toolkit';
import type {PayloadAction} from '@reduxjs/toolkit';

export interface MetricForecast {
    date: string;
    value: number;
}

export interface Forecast {
    id: string;
    date: string;
    revenue: number;
    roomSold: number;
    arrivalRoom: number;
    departureRoom: number;
    oooRoom: number;
}

interface ForecastState {
    forecasts: Forecast[];
    dateRangeForecasts: Forecast[];
    // Revenue Forecast
    revenue: MetricForecast[];
    // Room Sold Forecast
    roomSold: MetricForecast[];
    // Arrival Forecast
    arrival: MetricForecast[];
    // Departure Forecast
    departure: MetricForecast[];
    // OOO Forecast
    ooo: MetricForecast[];
    loading: boolean;
    error: string | null;
    count: number;
    availableDates: string[];
    minDate: string | null;
    maxDate: string | null;
}

const initialState: ForecastState = {
    forecasts: [],
    dateRangeForecasts: [],
    revenue: [],
    roomSold: [],
    arrival: [],
    departure: [],
    ooo: [],
    loading: false,
    error: null,
    count: 0,
    availableDates: [],
    minDate: null,
    maxDate: null,
};

const forecastSlice = createSlice({
    name: 'forecast',
    initialState,
    reducers: {
        // Get Available Dates
        getAvailableDatesStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getAvailableDatesSuccess: (state, action: PayloadAction<{ dates: string[]; minDate: string; maxDate: string }>) => {
            state.loading = false;
            state.availableDates = action.payload.dates;
            state.minDate = action.payload.minDate;
            state.maxDate = action.payload.maxDate;
            state.error = null;
        },
        getAvailableDatesFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Get Date Range Forecasts
        getDateRangeForecastsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDateRangeForecastsSuccess: (state, action: PayloadAction<{ forecasts: Forecast[]; count: number }>) => {
            state.loading = false;
            state.dateRangeForecasts = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getDateRangeForecastsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Download Forecast CSV
        downloadForecastCSVStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        downloadForecastCSVSuccess: (state, action: PayloadAction<{ forecasts: Forecast[]; count: number }>) => {
            state.loading = false;
            state.dateRangeForecasts = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        downloadForecastCSVFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Get Single Day Forecast
        getSingleDayForecastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getSingleDayForecastSuccess: (state) => {
            state.loading = false;
            state.error = null;
        },
        getSingleDayForecastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Get Forecasts by Period
        getForecastByPeriodStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getForecastByPeriodSuccess: (state, action: PayloadAction<{ forecasts: Forecast[]; count: number }>) => {
            state.loading = false;
            state.forecasts = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getForecastByPeriodFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Revenue Forecast Metrics
        getRevenueForecaastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRevenueForecaastSuccess: (state, action: PayloadAction<{ forecasts: MetricForecast[]; count: number }>) => {
            state.loading = false;
            state.revenue = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getRevenueForecaastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Room Sold Forecast Metrics
        getRoomSoldForecaastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRoomSoldForecaastSuccess: (state, action: PayloadAction<{ forecasts: MetricForecast[]; count: number }>) => {
            state.loading = false;
            state.roomSold = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getRoomSoldForecaastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Arrival Forecast Metrics
        getArrivalForecaastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getArrivalForecaastSuccess: (state, action: PayloadAction<{ forecasts: MetricForecast[]; count: number }>) => {
            state.loading = false;
            state.arrival = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getArrivalForecaastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Departure Forecast Metrics
        getDepartureForecaastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDepartureForecaastSuccess: (state, action: PayloadAction<{ forecasts: MetricForecast[]; count: number }>) => {
            state.loading = false;
            state.departure = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getDepartureForecaastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // OOO Forecast Metrics
        getOOOForecaastStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getOOOForecaastSuccess: (state, action: PayloadAction<{ forecasts: MetricForecast[]; count: number }>) => {
            state.loading = false;
            state.ooo = action.payload.forecasts;
            state.count = action.payload.count;
            state.error = null;
        },
        getOOOForecaastFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        resetForecasts: (state) => {
            state.forecasts = [];
            state.dateRangeForecasts = [];
            state.revenue = [];
            state.roomSold = [];
            state.arrival = [];
            state.departure = [];
            state.ooo = [];
            state.loading = false;
            state.error = null;
            state.count = 0;
            state.availableDates = [];
            state.minDate = null;
            state.maxDate = null;
        },
    },
});

export const {
    getAvailableDatesStart,
    getAvailableDatesSuccess,
    getAvailableDatesFailure,
    getDateRangeForecastsStart,
    getDateRangeForecastsSuccess,
    getDateRangeForecastsFailure,
    downloadForecastCSVStart,
    downloadForecastCSVSuccess,
    downloadForecastCSVFailure,
    getSingleDayForecastStart,
    getSingleDayForecastSuccess,
    getSingleDayForecastFailure,
    getForecastByPeriodStart,
    getForecastByPeriodSuccess,
    getForecastByPeriodFailure,
    // Revenue Forecast
    getRevenueForecaastStart,
    getRevenueForecaastSuccess,
    getRevenueForecaastFailure,
    // Room Sold Forecast
    getRoomSoldForecaastStart,
    getRoomSoldForecaastSuccess,
    getRoomSoldForecaastFailure,
    // Arrival Forecast
    getArrivalForecaastStart,
    getArrivalForecaastSuccess,
    getArrivalForecaastFailure,
    // Departure Forecast
    getDepartureForecaastStart,
    getDepartureForecaastSuccess,
    getDepartureForecaastFailure,
    // OOO Forecast
    getOOOForecaastStart,
    getOOOForecaastSuccess,
    getOOOForecaastFailure,
    resetForecasts,
} = forecastSlice.actions;

export default forecastSlice.reducer;
