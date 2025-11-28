import React, { useEffect, Suspense, lazy } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import AdminNavbar from '../components/dashboard/AdminNavbar';
import GeneralInfo from '../components/IndividualHotelPage/GeneralInfo';
import UserTable from '../components/IndividualHotelPage/UserTable';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import DownloadRecordCSV from '../components/IndividualHotelPage/DownloadRecordCSV';
import DownloadForcastCSV from '../components/IndividualHotelPage/DownloadForcastCSV';

// Lazy load graph components for better performance
const RevenueGraph = lazy(() => import('../components/IndividualHotelPage/RevenueGraph'));
const RoomSoldGraph = lazy(() => import('../components/IndividualHotelPage/RoomSoldGraph'));
const ArrivalRoomGraph = lazy(() => import('../components/IndividualHotelPage/ArrivalRoomGraph'));
const DepartureRoomGraph = lazy(() => import('../components/IndividualHotelPage/DepartureRoomGraph'));
const OOORoomGraph = lazy(() => import('../components/IndividualHotelPage/OOORoomGraph'));

const GraphSkeleton = () => (
    <div className="w-full h-[400px] bg-white rounded-xl shadow-sm border border-gray-100 p-6 animate-pulse">
        <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
        <div className="h-[300px] bg-gray-100 rounded"></div>
    </div>
);

const IndividualHotelPage: React.FC = () => {
    const { hotelId } = useParams<{ hotelId: string }>();
    const navigate = useNavigate();
    const userRole = localStorage.getItem('userRole') as 'admin' | 'owner' | 'manager' || 'admin';

    useEffect(() => {
        if (!hotelId) {
            navigate('/all-hotels');
        }
    }, [hotelId, navigate]);

    if (!hotelId) {
        return null;
    }

    return (
        <div className="min-h-screen bg-gray-50">
            <AdminNavbar role={userRole} />
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 inline-block">
                        Hotel Management
                    </h1>
                    <p className="mt-2 text-gray-600">Manage and monitor hotel performance metrics</p>
                </div>

                <div className="space-y-8">
                    {/* General Info Section */}
                    <GeneralInfo hotelId={hotelId} />

                    {/* Users Table Section */}
                    <UserTable hotelId={hotelId} />

                    {/* Recent Records Section */}
                    <RecentRecords hotelId={hotelId} />

                    {/* Downloads Section */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <DownloadRecordCSV hotelId={hotelId} />
                        <DownloadForcastCSV hotelId={hotelId} />
                    </div>

                    {/* Graphs Grid */}
                    <div className="grid grid-cols-1 gap-8">
                        <Suspense fallback={<GraphSkeleton />}>
                            <RevenueGraph hotelId={hotelId} />
                        </Suspense>
                        <Suspense fallback={<GraphSkeleton />}>
                            <RoomSoldGraph hotelId={hotelId} />
                        </Suspense>
                        <Suspense fallback={<GraphSkeleton />}>
                            <ArrivalRoomGraph hotelId={hotelId} />
                        </Suspense>
                        <Suspense fallback={<GraphSkeleton />}>
                            <DepartureRoomGraph hotelId={hotelId} />
                        </Suspense>
                        <Suspense fallback={<GraphSkeleton />}>
                            <OOORoomGraph hotelId={hotelId} />
                        </Suspense>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IndividualHotelPage;
