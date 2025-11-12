// src/pages/AdminDashboard.tsx
import { useEffect } from "react";
import AdminNavbar from "../components/dashboard/AdminNavbar.tsx";
import StatCard from "../components/dashboard/StatCard";
import HotelCard from "../components/dashboard/HotelCard";
import "../stylesheet/pages/page-dashboard.css";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { fetchDashboardStats, fetchRecentlyAddedHotels } from "../redux/services/api";

const AdminDashboard = () => {
    const dispatch = useAppDispatch();
    const { stats } = useAppSelector((state) => state.dashboard);
    const { recentHotels, loading: hotelsLoading } = useAppSelector((state) => state.hotels);

    useEffect(() => {
        // Fetch dashboard stats
        dispatch(fetchDashboardStats() as any);

        // Fetch recently added hotels (limit: 3)
        dispatch(fetchRecentlyAddedHotels(3) as any);
    }, [dispatch]);

    const mockImageUrl = "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=800&q=80";

    return (
        <div className="page-dashboard">
            <AdminNavbar role="admin" />

            <div className="dashboard-container">
                <h2>Master Admin Dashboard</h2>

                {/* Stats Section */}
                <div className="dashboard-stats">
                    <StatCard
                        title="Total Hotels"
                        value={stats?.totalHotels.toString() || "0"}
                        description="Registered properties"
                        icon="🏨"
                    />
                    <StatCard
                        title="Total Users"
                        value={stats?.totalUsers.toString() || "0"}
                        description="Property owners and managers"
                        icon="👤"
                    />
                    <StatCard
                        title="Active Models"
                        value="3"
                        description="ML models running"
                        icon="📈"
                    />
                </div>

                {/* Recent Hotels Section */}
                <div className="dashboard-hotels">
                    <div className="dashboard-header">
                        <h3>Recently Added Hotels</h3>
                        <button className="add-hotel-btn">Add New Hotel</button>
                    </div>

                    {hotelsLoading ? (
                        <div className="loading-message">Loading hotels...</div>
                    ) : (
                        <div className="hotel-cards-container">
                            {recentHotels.length > 0 ? (
                                recentHotels.map((hotel) => (
                                    <HotelCard
                                        key={hotel.id}
                                        name={hotel.hotelName}
                                        owner={hotel.ownerName}
                                        city={hotel.city}
                                        contact={hotel.contactNumber}
                                        imageUrl={hotel.imageUrl || mockImageUrl}
                                        onClick={() => console.log(`Clicked on ${hotel.hotelName}`)}
                                    />
                                ))
                            ) : (
                                <div className="no-hotels-message">No recently added hotels</div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AdminDashboard;