// filepath: c:\Btech\Sem 7\Revenue_Prediction_Website\frontend\src\pages\UserForecast.tsx
// import React from 'react';
import { useParams } from 'react-router-dom';
import { useSelector } from 'react-redux';
import type { RootState } from '../redux/store';
import UserNavbar from '../components/dashboard/UserNavbar';
import RevenueGraph from '../components/IndividualHotelPage/RevenueGraph';
import RoomSoldGraph from '../components/IndividualHotelPage/RoomSoldGraph';
import ArrivalRoomGraph from '../components/IndividualHotelPage/ArrivalRoomGraph';
import DepartureRoomGraph from '../components/IndividualHotelPage/DepartureRoomGraph';
import OOORoomGraph from '../components/IndividualHotelPage/OOORoomGraph';
import DownloadForcastCSV from '../components/IndividualHotelPage/DownloadForcastCSV';
import ForecastDayViewer from '../components/IndividualHotelPage/ForecastDayViewer';
import '../stylesheet/pages/page-user-forecast.css';

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
                                <RevenueGraph hotelId={hotelId} />
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <RoomSoldGraph hotelId={hotelId} />
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <ArrivalRoomGraph hotelId={hotelId} />
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <DepartureRoomGraph hotelId={hotelId} />
                            </div>
                            <div className="page-user-forecast-graph-wrapper">
                                <OOORoomGraph hotelId={hotelId} />
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
}

export default UserForecast;

