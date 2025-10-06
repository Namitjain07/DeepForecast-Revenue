// src/components/Navbar.tsx
import React from "react";
import "../../stylesheet/ui/components-ui-navbar.css";

interface NavbarProps {
    role: "admin" | "owner" | "manager";
}

const Navbar: React.FC<NavbarProps> = ({ role }) => {
    return (
        <nav className="components-ui-navbar">
            <div className="navbar-left">
                <div className="navbar-logo">🏨</div>
                <h2>Hotel Revenue Predictor</h2>
            </div>

            <div className="navbar-tabs">
                <button className="navbar-tab active">Dashboard</button>
                {role !== "manager" && <button className="navbar-tab">All Hotels</button>}
                {role === "admin" && <button className="navbar-tab">Add Hotel</button>}
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

export default Navbar;
