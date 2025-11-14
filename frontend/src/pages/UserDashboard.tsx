import { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import StatCard from '../components/dashboard/StatCard';
import { fetchHotelDashboardStats } from '../redux/services/hotelApi';
import { fetchRecentRecords } from '../redux/services';
import '../stylesheet/pages/page-user-dashboard.css';

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
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-dashboard">
                <div className="page-user-dashboard-header">
                    <h1>Dashboard</h1>
                    <p>Welcome back, {user?.name}! Here's your hotel's performance overview.</p>
                </div>

                {/* Stat Cards using StatCard Component */}
                <div className="page-user-dashboard-stats">
                    <StatCard
                        title="Total Revenue"
                        value={`₹${dashboardStats.totalRevenue.toLocaleString('en-IN')}`}
                        description="Last 30 days"
                        icon="💰"
                    />
                    <StatCard
                        title="Rooms Sold"
                        value={dashboardStats.totalRoomsSold}
                        description="Last 30 days"
                        icon="🛏️"
                    />
                    <StatCard
                        title="Occupancy Rate"
                        value={`${dashboardStats.avgOccupancyRate}%`}
                        description="Average rate for last 30 days"
                        icon="📊"
                    />
                </div>

                {/* Recent Records Section */}
                <div className="page-user-dashboard-recent-section">
                    <div className="page-user-dashboard-recent-section-title">Recent Records</div>
                    <RecentRecords hotelId={hotelId} />
                </div>
            </div>
        </div>
    );
}

export default UserDashboard;