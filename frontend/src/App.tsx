// import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Provider } from 'react-redux';
import { store } from './redux/store';
import Home from "./pages/Home";
import AdminDashboard from "./pages/AdminDashboard";
import OwnerDashboard from "./pages/OwnerDashboard";
import ManagerDashboard from "./pages/ManagerDashboard";
import AllHotels from "./pages/AllHotels";
import AddHotel from "./pages/AddHotel";
import { useAppSelector } from './redux/hooks';
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

function App() {
    return (
        <Provider store={store}>
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
                        path="/owner-dashboard"
                        element={
                            <PrivateRoute allowedRoles={['owner']}>
                                <OwnerDashboard />
                            </PrivateRoute>
                        }
                    />
                    <Route
                        path="/manager-dashboard"
                        element={
                            <PrivateRoute allowedRoles={['manager']}>
                                <ManagerDashboard />
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
                </Routes>
            </BrowserRouter>
        </Provider>
    );
}

export default App;