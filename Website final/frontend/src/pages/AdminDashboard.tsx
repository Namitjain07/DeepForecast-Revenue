// src/pages/AdminDashboard.tsx
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AdminNavbar from "../components/dashboard/AdminNavbar.tsx";
import StatCard from "../components/dashboard/StatCard";
import HotelCard from "../components/dashboard/HotelCard";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { fetchDashboardStats, fetchRecentlyAddedHotels } from "../redux/services/api";

const AdminDashboard = () => {
    const dispatch = useAppDispatch();
    const navigate = useNavigate();
    const { stats } = useAppSelector((state) => state.dashboard);
    const { recentHotels, loading: hotelsLoading } = useAppSelector((state) => state.hotels);

    useEffect(() => {
        // Fetch dashboard stats
        dispatch(fetchDashboardStats());

        // Fetch recently added hotels (limit: 3)
        dispatch(fetchRecentlyAddedHotels(3));
    }, [dispatch]);

    const handleHotelClick = (hotelId: string) => {
        navigate(`/hotel/${hotelId}`);
    };

    const mockImageUrl = "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?auto=format&fit=crop&w=800&q=80";

    return (
        <div className="min-h-screen bg-gray-50">
            <AdminNavbar role="admin" />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">Master Admin Dashboard</h1>
                    <p className="mt-1 text-sm text-gray-500">Overview of system performance and recent activities</p>
                </div>

                {/* Stats Section */}
                <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 mb-8">
                    <StatCard
                        title="Total Hotels"
                        value={stats?.totalHotels.toString() || "0"}
                        description="Registered properties"
                        icon="🏨"
                        color="indigo"
                    />
                    <StatCard
                        title="Total Users"
                        value={stats?.totalUsers.toString() || "0"}
                        description="Property owners and managers"
                        icon="👤"
                        color="blue"
                    />
                    <StatCard
                        title="Active Models"
                        value="3"
                        description="ML models running"
                        icon="📈"
                        color="green"
                    />
                </div>

                {/* Recent Hotels Section */}
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="px-6 py-5 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                        <h3 className="text-lg font-medium leading-6 text-gray-900">Recently Added Hotels</h3>
                        <button 
                            onClick={() => navigate('/add-hotel')}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors"
                        >
                            Add New Hotel
                        </button>
                    </div>

                    <div className="p-6">
                        {hotelsLoading ? (
                            <div className="flex justify-center items-center h-32 text-gray-500">
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-indigo-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Loading hotels...
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
                                {recentHotels.length > 0 ? (
                                    recentHotels.map((hotel) => (
                                        <HotelCard
                                            key={hotel.id}
                                            name={hotel.hotelName}
                                            owner={hotel.ownerName}
                                            city={hotel.city}
                                            contact={hotel.contactNumber}
                                            imageUrl={hotel.imageUrl || mockImageUrl}
                                            onClick={() => handleHotelClick(hotel.id)}
                                        />
                                    ))
                                ) : (
                                    <div className="col-span-full text-center py-12 text-gray-500 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
                                        No recently added hotels found
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AdminDashboard;