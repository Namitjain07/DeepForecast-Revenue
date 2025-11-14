import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import AdminNavbar from '../components/dashboard/AdminNavbar';
import GeneralInfo from '../components/IndividualHotelPage/GeneralInfo';
import UserTable from '../components/IndividualHotelPage/UserTable';
import RecentRecords from '../components/IndividualHotelPage/RecentRecords';
import DownloadRecordCSV from '../components/IndividualHotelPage/DownloadRecordCSV.tsx';
import DownloadForcastCSV from '../components/IndividualHotelPage/DownloadForcastCSV.tsx';
import RevenueGraph from '../components/IndividualHotelPage/RevenueGraph';
import RoomSoldGraph from '../components/IndividualHotelPage/RoomSoldGraph';
import ArrivalRoomGraph from '../components/IndividualHotelPage/ArrivalRoomGraph';
import DepartureRoomGraph from '../components/IndividualHotelPage/DepartureRoomGraph';
import OOORoomGraph from '../components/IndividualHotelPage/OOORoomGraph';
import '../stylesheet/pages/page-individual-hotel.css';

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
        <div>
            <AdminNavbar role={userRole} />
            <div className="page-individual-hotel-container">
                <div className="page-individual-hotel-header">
                    <h1>Hotel Management</h1>
                </div>

                <div className="page-individual-hotel-content">
                    <section className="page-individual-hotel-section">
                        <GeneralInfo hotelId={hotelId} />
                    </section>

                    <section className="page-individual-hotel-section">
                        <UserTable hotelId={hotelId} />
                    </section>

                    <section className="page-individual-hotel-section">
                        <RecentRecords hotelId={hotelId} />
                    </section>

                    <section className="page-individual-hotel-section">
                        <DownloadRecordCSV hotelId={hotelId} />
                    </section>

                    <section className="page-individual-hotel-section">
                        <DownloadForcastCSV hotelId={hotelId} />
                    </section>

                    <div className="page-individual-hotel-graphs-grid">
                        <section className="page-individual-hotel-section">
                            <RevenueGraph hotelId={hotelId} />
                        </section>

                        <section className="page-individual-hotel-section">
                            <RoomSoldGraph hotelId={hotelId} />
                        </section>

                        <section className="page-individual-hotel-section">
                            <ArrivalRoomGraph hotelId={hotelId} />
                        </section>

                        <section className="page-individual-hotel-section">
                            <DepartureRoomGraph hotelId={hotelId} />
                        </section>

                        <section className="page-individual-hotel-section">
                            <OOORoomGraph hotelId={hotelId} />
                        </section>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default IndividualHotelPage;
