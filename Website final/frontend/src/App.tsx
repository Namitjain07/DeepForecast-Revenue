// import React from 'react';
import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Provider, useDispatch } from 'react-redux';
import { store } from './redux/store';
import Home from "./pages/Home";
import AdminDashboard from "./pages/AdminDashboard";
import UserDashboard from "./pages/UserDashboard.tsx";
import AllHotels from "./pages/AllHotels";
import AddHotel from "./pages/AddHotel";
import IndividualHotelPage from "./pages/IndividualHotelPage";
import UserForecast from "./pages/UserForecast.tsx";
import UserHotelInfo from "./pages/UserHotelInfo.tsx";
import UserManager from "./pages/UserManager.tsx";
import UserRecords from "./pages/UserRecords.tsx";
import { useAppSelector } from './redux/hooks';
import { restoreAuthState } from './redux/slices/authSlice';
import type {JSX} from "react";

const PrivateRoute = ({ children, allowedRoles }: { children: JSX.Element, allowedRoles: string[] }) => {
    const { isAuthenticated, user } = useAppSelector((state) => state.auth);

    if (!isAuthenticated) {
        return <Navigate to="/" />;
    }

    // @ts-ignore
    if (!user || !allowedRoles.includes(user.role)) {
        return <Navigate to="/" />;
    }

    return children;
};

// Wrapper component to handle auth restoration
const AppContent = () => {
    const dispatch = useDispatch();

    useEffect(() => {
        // Restore auth state from localStorage on app mount
        dispatch(restoreAuthState());
    }, [dispatch]);

    return (
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route
                    path="/admin-dashboard"
                    element={
                        <PrivateRoute allowedRoles={['admin']}>
                            <AdminDashboard />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/user-dashboard"
                    element={
                        <PrivateRoute allowedRoles={['owner','manager']}>
                            <UserDashboard />
                        </PrivateRoute>
                    }
                />

                <Route
                    path="/all-hotels"
                    element={
                        <PrivateRoute allowedRoles={['admin', 'owner']}>
                            <AllHotels />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/add-hotel"
                    element={
                        <PrivateRoute allowedRoles={['admin']}>
                            <AddHotel />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/hotel/:hotelId"
                    element={
                        <PrivateRoute allowedRoles={['admin', 'owner', 'manager']}>
                            <IndividualHotelPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/hotel-info/:hotelId"
                    element={
                        <PrivateRoute allowedRoles={['owner', 'manager']}>
                            <UserHotelInfo />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/forecast/:hotelId"
                    element={
                        <PrivateRoute allowedRoles={['owner', 'manager']}>
                            <UserForecast />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/manager/:hotelId"
                    element={
                        <PrivateRoute allowedRoles={['owner']}>
                            <UserManager />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/records/:hotelId"
                    element={
                        <PrivateRoute allowedRoles={['owner', 'manager']}>
                            <UserRecords />
                        </PrivateRoute>
                    }
                />
            </Routes>
        </BrowserRouter>
    );
};

function App() {
    return (
        <Provider store={store}>
            <AppContent />
        </Provider>
    );
}

export default App;