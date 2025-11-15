import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';

interface LastTrain {
    id: string;
    hotelId: string;
    userId: string;
    startDateTime: string;
    endDateTime: string;
    status: 'none' | 'queued' | 'running' | 'success' | 'failure';
    createdAt: string;
    updatedAt: string;
}

interface ModelTrainState {
    lastTrain: LastTrain | null;
    loading: boolean;
    error: string | null;
    success: boolean;
}

const initialState: ModelTrainState = {
    lastTrain: null,
    loading: false,
    error: null,
    success: false,
};

const modelTrainSlice = createSlice({
    name: 'modelTrain',
    initialState,
    reducers: {
        // Add Last Train
        addLastTrainStart: (state) => {
            state.loading = true;
            state.error = null;
            state.success = false;
        },
        addLastTrainSuccess: (state, action: PayloadAction<LastTrain>) => {
            state.loading = false;
            state.lastTrain = action.payload;
            state.error = null;
            state.success = true;
        },
        addLastTrainFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
            state.success = false;
        },

        // Get Last Train by Hotel
        getLastTrainStart: (state) => {
            state.loading = true;
            state.error = null;
        },
        getLastTrainSuccess: (state, action: PayloadAction<LastTrain>) => {
            state.loading = false;
            state.lastTrain = action.payload;
            state.error = null;
        },
        getLastTrainFailure: (state, action: PayloadAction<string>) => {
            state.loading = false;
            state.error = action.payload;
        },

        // Reset state
        resetModelTrainState: (state) => {
            state.lastTrain = null;
            state.loading = false;
            state.error = null;
            state.success = false;
        },
    },
});

export const {
    addLastTrainStart,
    addLastTrainSuccess,
    addLastTrainFailure,
    getLastTrainStart,
    getLastTrainSuccess,
    getLastTrainFailure,
    resetModelTrainState,
} = modelTrainSlice.actions;

export default modelTrainSlice.reducer;

