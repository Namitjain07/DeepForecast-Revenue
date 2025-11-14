import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchRecentRecords } from '../../redux/services/api';
import '../../stylesheet/ui/component-ui-recent-record.css';

interface RecentRecordsProps {
    hotelId?: string;
}

const RecentRecords: React.FC<RecentRecordsProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { recentRecords, loading, error } = useSelector(
        (state: RootState) => state.records
    );

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchRecentRecords(hotelId) as any);
        }
    }, [hotelId, dispatch]);

    const formatDate = (dateString: string) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    if (loading) {
        return <div className="component-ui-recent-record-loading">Loading records...</div>;
    }

    if (error) {
        return <div className="component-ui-recent-record-error">Error: {error}</div>;
    }

    if (!recentRecords || recentRecords.length === 0) {
        return <div className="component-ui-recent-record-empty">No recent records found</div>;
    }

    return (
        <div className="component-ui-recent-record-container">
            <h2 className="component-ui-recent-record-title">Recent Records</h2>
            <div className="component-ui-recent-record-wrapper">
                <table className="component-ui-recent-record-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Day</th>
                            <th>Rooms Sold</th>
                            <th>Arrival Rooms</th>
                            <th>Departure Rooms</th>
                            <th>Occupancy %</th>
                            <th>Room Revenue</th>
                            <th>Avg Room Rate</th>
                            <th>PAX</th>
                            <th>Total Inventory</th>
                            <th>OOO Rooms</th>
                            <th>Compliment Rooms</th>
                            <th>House Use</th>
                            <th>Individual Confirm</th>
                        </tr>
                    </thead>
                    <tbody>
                        {recentRecords.map(record => (
                            <tr key={record.id}>
                                <td>{formatDate(record.date)}</td>
                                <td>{record.day}</td>
                                <td>{record.roomsSold}</td>
                                <td>{record.arrivalRooms}</td>
                                <td>{record.departureRooms}</td>
                                <td>{record.occupancyPercentage.toFixed(2)}%</td>
                                <td>${record.roomRevenue}</td>
                                <td>${record.averageRoomRate.toFixed(2)}</td>
                                <td>{record.pax}</td>
                                <td>{record.totalRoomInventory}</td>
                                <td>{record.oooRooms}</td>
                                <td>{record.complimentRooms}</td>
                                <td>{record.houseUse}</td>
                                <td>{record.individualConfirm}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default RecentRecords;

