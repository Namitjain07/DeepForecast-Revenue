import { configureStore } from "@reduxjs/toolkit";
import counterReducer from "./slices/counterSlice.ts";
import { api } from "./services/api";

export const store = configureStore({
    reducer: {
        counter: counterReducer,
        [api.reducerPath]: api.reducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware().concat(api.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
