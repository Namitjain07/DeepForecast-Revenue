import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import StatCard from '../components/dashboard/StatCard';
import { fetchHotelDashboardStats } from '../redux/services/hotelApi';
import { fetchRecentRecords } from '../redux/services';

function UserDashboard() {
    const { user } = useSelector((state: RootState) => state.auth);

    const dispatch = useDispatch<AppDispatch>();
    const role = user?.role as 'owner' | 'manager' || 'manager';

    // Get hotel ID from user data
    // @ts-ignore
    const hotelId = user?.hotelId || '';

    const [dashboardStats, setDashboardStats] = useState({
        totalRevenue: 0,
        totalRoomsSold: 0,
        avgOccupancyRate: 0,
        loading: true,
        error: null as string | null
    });

    useEffect(() => {
        if (hotelId) {
            // Fetch both dashboard stats and recent records
            const fetchStats = async () => {
                try {
                    const stats = await dispatch(fetchHotelDashboardStats(hotelId) as any);
                    if (stats) {
                        setDashboardStats({
                            totalRevenue: stats.totalRevenue || 0,
                            totalRoomsSold: stats.totalRoomsSold || 0,
                            avgOccupancyRate: stats.avgOccupancyRate || 0,
                            loading: false,
                            error: null
                        });
                    }
                } catch (error: any) {
                    setDashboardStats(prev => ({
                        ...prev,
                        loading: false,
                        error: error.message || 'Failed to fetch stats'
                    }));
                }
            };

            fetchStats();
            dispatch(fetchRecentRecords(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    return (
        <div className="min-h-screen bg-gray-50">
            <UserNavbar role={role} hotelId={hotelId} />
            
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
                    <p className="mt-1 text-sm text-gray-500">
                        Welcome back, {user?.name}! Here's your hotel's performance overview.
                    </p>
                </div>

                {/* Stat Cards using StatCard Component */}
                <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 mb-8">
                    <StatCard
                        title="Total Revenue"
                        value={`₹${dashboardStats.totalRevenue.toLocaleString('en-IN')}`}
                        description="Last 30 days"
                        icon="💰"
                        color="green"
                    />
                    <StatCard
                        title="Rooms Sold"
                        value={dashboardStats.totalRoomsSold}
                        description="Last 30 days"
                        icon="🛏️"
                        color="blue"
                    />
                    <StatCard
                        title="Occupancy Rate"
                        value={`${dashboardStats.avgOccupancyRate}%`}
                        description="Average rate for last 30 days"
                        icon="📊"
                        color="purple"
                    />
                </div>

                {/* Recent Records Section */}
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
                    <div className="px-6 py-5 border-b border-gray-100 bg-gray-50/50">
                        <h3 className="text-lg font-medium leading-6 text-gray-900">Recent Records</h3>
                    </div>
                    <div className="p-6">
                        <RecentRecords hotelId={hotelId} />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default UserDashboard;