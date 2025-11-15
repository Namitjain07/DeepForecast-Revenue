import { configureStore } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import hotelReducer from './slices/hotelSlice';
import dashboardReducer from './slices/dashboardSlice';
import usersReducer from './slices/usersSlice';
import recordsReducer from './slices/recordsSlice';
import forecastReducer from './slices/forcastSlice';
import modelTrainReducer from './slices/modelTrainSlice';

export const store = configureStore({
    reducer: {
        auth: authReducer,
        hotels: hotelReducer,
        dashboard: dashboardReducer,
        users: usersReducer,
        records: recordsReducer,
        forecast: forecastReducer,
        modelTrain: modelTrainReducer,
    },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
