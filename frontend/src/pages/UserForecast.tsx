// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserForecast.tsx
import { Suspense, lazy } from 'react';
import { useParams } from 'react-router-dom';
import { useSelector } from 'react-redux';
import type { RootState } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import DownloadForcastCSV from '../components/IndividualHotelPage/DownloadForcastCSV';
import ForecastDayViewer from '../components/IndividualHotelPage/ForecastDayViewer';
import '../stylesheet/pages/page-user-forecast.css';

const RevenueGraph = lazy(() => import('../components/IndividualHotelPage/RevenueGraph'));
const RoomSoldGraph = lazy(() => import('../components/IndividualHotelPage/RoomSoldGraph'));
const ArrivalRoomGraph = lazy(() => import('../components/IndividualHotelPage/ArrivalRoomGraph'));
const DepartureRoomGraph = lazy(() => import('../components/IndividualHotelPage/DepartureRoomGraph'));
const OOORoomGraph = lazy(() => import('../components/IndividualHotelPage/OOORoomGraph'));

const GraphSkeleton = () => (
    <div className="w-full h-[400px] bg-gray-100 rounded animate-pulse flex items-center justify-center text-gray-400">
        Loading Graph...
    </div>
);

function UserForecast() {
    const { hotelId } = useParams<{ hotelId: string }>();
    const { user } = useSelector((state: RootState) => state.auth);
    const role = user?.role as 'owner' | 'manager' || 'manager';

    return (
        <div>
            <UserNavbar role={role} hotelId={hotelId} />
            <div className="page-user-forecast">
                <div className="page-user-forecast-header">
                    <h1>Forecast Analysis</h1>
                    <p>View revenue and room forecasts for your hotel across different time periods.</p>
                </div>

                {/* Download Forecast CSV Section */}
                {hotelId && (
                    <div className="page-user-forecast-download-section">
                        <DownloadForcastCSV hotelId={hotelId} />
                    </div>
                )}

                {/* Forecast Day Viewer Section */}
                {hotelId && (
                    <div className="page-user-forecast-day-viewer-section">
                        <ForecastDayViewer hotelId={hotelId} />
                    </div>
                )}

                {/* Forecast Graphs */}
                <div className="page-user-forecast-container">
                    {hotelId && (
                        <>
                            <div className="page-user-forecast-graph-wrapper">
                                <Suspense fallback={<GraphSkeleton />}>
                                    <RevenueGraph hotelId={hotelId} />
                                </Suspense>
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <Suspense fallback={<GraphSkeleton />}>
                                    <RoomSoldGraph hotelId={hotelId} />
                                </Suspense>
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <Suspense fallback={<GraphSkeleton />}>
                                    <ArrivalRoomGraph hotelId={hotelId} />
                                </Suspense>
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <Suspense fallback={<GraphSkeleton />}>
                                    <DepartureRoomGraph hotelId={hotelId} />
                                </Suspense>
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <Suspense fallback={<GraphSkeleton />}>
                                    <OOORoomGraph hotelId={hotelId} />
                                </Suspense>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

export default UserForecast;

