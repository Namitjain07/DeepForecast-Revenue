// src/pages/AdminDashboard.tsx
// import React from "react";
// import { Hotel, Users, TrendingUp } from "lucide-react";
import Navbar from "../components/dashboard/Navbar";
import StatCard from "../components/dashboard/StatCard";
import HotelCard from "../components/dashboard/HotelCard";
import "../stylesheet/pages/page-dashboard.css";

const hotels = [
    {
        name: "Grand Hotel Mumbai",
        owner: "John Smith",
        city: "Mumbai",
        contact: "+91 98765 43210",
        imageUrl: "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=800&q=80"
    },
    {
        name: "Royal Palace Delhi",
        owner: "Priya Sharma",
        city: "Delhi",
        contact: "+91 98765 43211",
        imageUrl: "https://images.unsplash.com/photo-1571896349842-33c89424de2d?auto=format&fit=crop&w=800&q=80"
    },
    {
        name: "Seaside Resort Goa",
        owner: "Carlos D'Silva",
        city: "Goa",
        contact: "+91 98765 43212",
        imageUrl: "https://images.unsplash.com/photo-1520250497591-112f2f40a3f4?auto=format&fit=crop&w=800&q=80"
    },
];

const AdminDashboard = () => {
    return (
        <div className="page-dashboard">
            <Navbar role="admin" />

            <div className="dashboard-container">
                <h2>Master Admin Dashboard</h2>

                <div className="dashboard-stats">
                    <StatCard title="Total Hotels" value="3" description="Registered properties" icon="🏨" />
                    <StatCard title="Total Owners" value="3" description="Property owners" icon="👤" />
                    <StatCard title="Active Models" value="3" description="ML models running" icon="📈" />
                </div>

                <div className="dashboard-hotels">
                    <div className="dashboard-header">
                        <h3>All Hotels</h3>
                        <button className="add-hotel-btn">Add New Hotel</button>
                    </div>

                    <div className="hotel-cards-container">
                        {hotels.map((hotel, idx) => (
                            <HotelCard
                                key={idx}
                                {...hotel}
                                onClick={() => alert(`Clicked on ${hotel.name}`)}
                            />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdminDashboard;