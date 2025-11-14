// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\components\dashboard\UserNavbar.tsx
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import type { RootState } from "../../redux/store";
import { logout } from "../../redux/slices/authSlice";
import "../../stylesheet/ui/component-ui-user-navbar.css";

interface UserNavbarProps {
    role: "owner" | "manager";
    hotelId?: string;
}

const UserNavbar: React.FC<UserNavbarProps> = ({ role, hotelId }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const dispatch = useDispatch();
    const { user } = useSelector((state: RootState) => state.auth);
    const [activeTab, setActiveTab] = useState("Dashboard");
    const [showProfileDialog, setShowProfileDialog] = useState(false);

    const handleNavigation = (tab: string, path: string) => {
        setActiveTab(tab);
        navigate(path);
    };

    const handleLogout = () => {
        dispatch(logout());
        setShowProfileDialog(false);
        navigate("/");
    };

    // Automatically update active tab when user navigates manually
    useEffect(() => {
        if (location.pathname.includes("dashboard")) setActiveTab("Dashboard");
        else if (location.pathname.includes("hotel-info")) setActiveTab("Hotel Info");
        else if (location.pathname.includes("forecast")) setActiveTab("Forecast");
        else if (location.pathname.includes("manager")) setActiveTab("Manager");
        else if (location.pathname.includes("records")) setActiveTab("Records");
    }, [location.pathname]);

    // Get first letter of name for avatar
    const getInitial = () => {
        return user?.name ? user.name.charAt(0).toUpperCase() : "U";
    };

    return (
        <nav className="component-ui-user-navbar">
            <div className="user-navbar-left">
                <div className="user-navbar-logo">🏨</div>
                <h2>Hotel Revenue Predictor</h2>
            </div>

            <div className="user-navbar-tabs">
                {/* Dashboard - visible to both owner and manager */}
                <button
                    className={`user-navbar-tab ${activeTab === "Dashboard" ? "active" : ""}`}
                    onClick={() => handleNavigation("Dashboard", `/user-dashboard`)}
                >
                    Dashboard
                </button>

                {/* Hotel Info - visible to both owner and manager */}
                <button
                    className={`user-navbar-tab ${activeTab === "Hotel Info" ? "active" : ""}`}
                    onClick={() => handleNavigation("Hotel Info", `/hotel-info/${hotelId}`)}
                >
                    Hotel Info
                </button>

                {/* Forecast - visible to both owner and manager */}
                <button
                    className={`user-navbar-tab ${activeTab === "Forecast" ? "active" : ""}`}
                    onClick={() => handleNavigation("Forecast", `/forecast/${hotelId}`)}
                >
                    Forecast
                </button>

                {/* Manager - visible only to owner */}
                {role === "owner" && (
                    <button
                        className={`user-navbar-tab ${activeTab === "Manager" ? "active" : ""}`}
                        onClick={() => handleNavigation("Manager", `/manager/${hotelId}`)}
                    >
                        Manager
                    </button>
                )}

                {/* Records - visible to both owner and manager */}
                <button
                    className={`user-navbar-tab ${activeTab === "Records" ? "active" : ""}`}
                    onClick={() => handleNavigation("Records", `/records/${hotelId}`)}
                >
                    Records
                </button>
            </div>

            <div className="user-navbar-right">
                {/* Retrain Model Button - visible to both owner and manager */}
                <button className="user-navbar-button">Retrain Model</button>

                {/* Avatar */}
                <div
                    className="user-navbar-avatar"
                    onClick={() => setShowProfileDialog(!showProfileDialog)}
                    style={{ cursor: "pointer" }}
                >
                    {getInitial()}
                </div>

                {/* Profile Dialog */}
                {showProfileDialog && (
                    <div className="user-navbar-profile-dialog">
                        <div className="user-profile-dialog-content">
                            <div className="user-profile-header">
                                <div className="user-profile-avatar-large">{getInitial()}</div>
                                <h3>{user?.name || "User"}</h3>
                            </div>

                            <div className="user-profile-info">
                                <div className="user-info-item">
                                    <span className="user-info-label">Email:</span>
                                    <span className="user-info-value">{user?.email || "N/A"}</span>
                                </div>
                                <div className="user-info-item">
                                    <span className="user-info-label">Role:</span>
                                    <span className="user-info-value">{user?.role || role}</span>
                                </div>
                            </div>

                            <button
                                className="user-profile-logout-btn"
                                onClick={handleLogout}
                            >
                                Logout
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </nav>
    );
};

export default UserNavbar;

