// src/components/AdminNavbar.tsx
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import type { RootState } from "../../redux/store";
import { logout } from "../../redux/slices/authSlice";
import "../../stylesheet/ui/components-ui-navbar.css";

interface NavbarProps {
    role: "admin" | "owner" | "manager";
}

const AdminNavbar: React.FC<NavbarProps> = ({ role }) => {
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
        navigate("/login");
    };

    // Automatically update active tab when user navigates manually (e.g., using browser back button)
    useEffect(() => {
        if (location.pathname.includes("dashboard")) setActiveTab("Dashboard");
        else if (location.pathname.includes("all-hotels")) setActiveTab("All Hotels");
        else if (location.pathname.includes("add-hotel")) setActiveTab("Add Hotel");
    }, [location.pathname]);

    // Get first letter of name for avatar
    const getInitial = () => {
        return user?.name ? user.name.charAt(0).toUpperCase() : "A";
    };

    return (
        <nav className="components-ui-navbar">
            <div className="navbar-left">
                <div className="navbar-logo">🏨</div>
                <h2>Hotel Revenue Predictor</h2>
            </div>

            <div className="navbar-tabs">
                <button
                    className={`navbar-tab ${activeTab === "Dashboard" ? "active" : ""}`}
                    onClick={() => handleNavigation("Dashboard", `/${role}-dashboard`)}
                >
                    Dashboard
                </button>

                {role === "admin" && (
                    <button
                        className={`navbar-tab ${activeTab === "All Hotels" ? "active" : ""}`}
                        onClick={() => handleNavigation("All Hotels", `/all-hotels`)}
                    >
                        All Hotels
                    </button>
                )}

                {role === "admin" && (
                    <button
                        className={`navbar-tab ${activeTab === "Add Hotel" ? "active" : ""}`}
                        onClick={() => handleNavigation("Add Hotel", `/add-hotel`)}
                    >
                        Add Hotel
                    </button>
                )}
            </div>

            <div className="navbar-right">
                {role === "owner" && (
                    <button className="navbar-button">Retrain Model</button>
                )}
                <div
                    className="navbar-avatar"
                    onClick={() => setShowProfileDialog(!showProfileDialog)}
                    style={{ cursor: "pointer" }}
                >
                    {getInitial()}
                </div>

                {/* Profile Dialog */}
                {showProfileDialog && (
                    <div className="navbar-profile-dialog">
                        <div className="profile-dialog-content">
                            <div className="profile-header">
                                <div className="profile-avatar-large">{getInitial()}</div>
                                <h3>{user?.name || "User"}</h3>
                            </div>

                            <div className="profile-info">
                                <div className="info-item">
                                    <span className="info-label">Email:</span>
                                    <span className="info-value">{user?.email || "N/A"}</span>
                                </div>
                                <div className="info-item">
                                    <span className="info-label">Role:</span>
                                    <span className="info-value">{user?.role || role}</span>
                                </div>
                            </div>

                            <button
                                className="profile-logout-btn"
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

export default AdminNavbar;
