import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

export interface Hotel {
    id: string;
    name: string;
    email: string;
    contactNumber: string;
    city: string;
    state: string;
    imageUrl?: string;
    adminName?: string;
    ownerName: string;
}

export interface HotelGeneralInfo {
    id: string;
    name: string;
    email: string;
    contactNumber: string;
    plotNo: string;
    streetName: string;
    city: string;
    state: string;
    pincode: string;
}

export interface RecentHotel {
    id: string;
    hotelName: string;
    ownerName: string;
    city: string;
    contactNumber: string;
    imageUrl?: string;
    addedAt?: string;
}

export interface PaginationInfo {
    currentPage: number;
    pageSize: number;
    totalCount: number;
    totalPages: number;
    hasNextPage: boolean;
    hasPreviousPage?: boolean;
}

interface HotelState {
    hotels: Hotel[];
    recentHotels: RecentHotel[];
    pagination: PaginationInfo | null;
    loading: boolean;
    error: string | null;
    searchResults: Hotel[];
    searchPagination: PaginationInfo | null;
    generalInfo: HotelGeneralInfo | null;
}

const initialState: HotelState = {
    hotels: [],
    recentHotels: [],
    pagination: null,
    loading: false,
    error: null,
    searchResults: [],
    searchPagination: null,
    generalInfo: null,
};

const hotelSlice = createSlice({
    name: 'hotels',
    initialState,
    reducers: {
        // All Hotels
        getAllHotelsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getAllHotelsSuccess: (state, action: PayloadAction<{ hotels: Hotel[]; pagination: PaginationInfo }>) => {
            state.loading = false;
            state.hotels = action.payload.hotels;
            state.pagination = action.payload.pagination;
        },
        getAllHotelsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Append Hotels (for infinite scroll)
        appendHotels: (state, action: PayloadAction<{ hotels: Hotel[]; pagination: PaginationInfo }>) => {
            state.hotels = [...state.hotels, ...action.payload.hotels];
            state.pagination = action.payload.pagination;
            state.loading = false;
        },

        // Recently Added Hotels
        getRecentHotelsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getRecentHotelsSuccess: (state, action: PayloadAction<RecentHotel[]>) => {
            state.loading = false;
            state.recentHotels = action.payload;
        },
        getRecentHotelsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Search Hotels
        searchHotelsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        searchHotelsSuccess: (state, action: PayloadAction<{ hotels: Hotel[]; pagination: PaginationInfo }>) => {
            state.loading = false;
            state.searchResults = action.payload.hotels;
            state.searchPagination = action.payload.pagination;
        },
        searchHotelsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Clear search
        clearSearch: (state) => {
            state.searchResults = [];
            state.searchPagination = null;
            state.error = null;
        },

        // Clear error
        clearError: (state) => {
            state.error = null;
        },

        // General Info
        getGeneralInfoStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getGeneralInfoSuccess: (state, action: PayloadAction<HotelGeneralInfo>) => {
            state.loading = false;
            state.generalInfo = action.payload;
            state.error = null;
        },
        getGeneralInfoFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },
        resetGeneralInfo: (state) => {
            state.generalInfo = null;
            state.loading = false;
            state.error = null;
        },

        // Add Hotel
        addHotelStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        addHotelSuccess: (state, action: PayloadAction<Hotel>) => {
            state.loading = false;
            state.hotels.push(action.payload);
            state.error = null;
        },
        addHotelFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Fetch Hotel Dashboard Stats
        getDashboardStatsStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getDashboardStatsSuccess: (state) => {
            state.loading = false;
            state.error = null;
        },
        getDashboardStatsFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Update Hotel Info
        updateHotelInfoStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        updateHotelInfoSuccess: (state, action: PayloadAction<HotelGeneralInfo>) => {
            state.loading = false;
            state.generalInfo = action.payload;
            state.error = null;
        },
        updateHotelInfoFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },
    },
});

export const {
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
    clearSearch,
    clearError,
    getGeneralInfoStart,
    getGeneralInfoSuccess,
    getGeneralInfoFailure,
    resetGeneralInfo,
    addHotelStart,
    addHotelSuccess,
    addHotelFailure,
    getDashboardStatsStart,
    getDashboardStatsSuccess,
    getDashboardStatsFailure,
    updateHotelInfoStart,
    updateHotelInfoSuccess,
    updateHotelInfoFailure,
} = hotelSlice.actions;

export default hotelSlice.reducer;

