// src/components/AdminNavbar.tsx
import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "../../stylesheet/ui/components-ui-navbar.css";

interface NavbarProps {
    role: "admin" | "owner" | "manager";
}

const AdminNavbar: React.FC<NavbarProps> = ({ role }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [activeTab, setActiveTab] = useState("Dashboard");

    const handleNavigation = (tab: string, path: string) => {
        setActiveTab(tab);
        navigate(path);
    };

    // Automatically update active tab when user navigates manually (e.g., using browser back button)
    useEffect(() => {
        if (location.pathname.includes("dashboard")) setActiveTab("Dashboard");
        else if (location.pathname.includes("all-hotels")) setActiveTab("All Hotels");
        else if (location.pathname.includes("add-hotel")) setActiveTab("Add Hotel");
    }, [location.pathname]);

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
                {role === "admin" && (
                    <button className="navbar-button">Retrain Model</button>
                )}
                <div className="navbar-avatar">A</div>
            </div>
        </nav>
    );
};

export default AdminNavbar;
