import React, { useEffect, useState } from 'react';
import '../../stylesheet/ui/component-ui-recent-record.css';

interface RecentRecordsProps {
    hotelId: string;
}

interface Record {
    id: string;
    date: string;
    roomsSold: number;
    day: string;
    arrivalRooms: number;
    complimentRooms: number;
    houseUse: number;
    individualConfirm: number;
    occupancyPercentage: number;
    roomRevenue: number;
    averageRoomRate: number;
    departureRooms: number;
    oooRooms: number;
    pax: number;
    totalRoomInventory: number;
}

const RecentRecords: React.FC<RecentRecordsProps> = ({ hotelId }) => {
    const [records, setRecords] = useState<Record[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Fetch records for this hotel
        const fetchRecords = async () => {
            try {
                setLoading(true);
                // Replace with your actual API call
                // const response = await fetch(`/api/hotels/${hotelId}/records`);
                // const data = await response.json();
                // setRecords(data);

                // Temporary mock data
                setRecords([
                    {
                        id: '1',
                        date: '2025-08-14T18:34:46.908+00:00',
                        roomsSold: 25,
                        day: 'Friday',
                        arrivalRooms: 60,
                        complimentRooms: 2,
                        houseUse: 2,
                        individualConfirm: 9,
                        occupancyPercentage: 18.94,
                        roomRevenue: 2575,
                        averageRoomRate: 394.07,
                        departureRooms: 21,
                        oooRooms: 2,
                        pax: 25,
                        totalRoomInventory: 132
                    },
                    {
                        id: '2',
                        date: '2025-08-13T18:34:46.908+00:00',
                        roomsSold: 42,
                        day: 'Thursday',
                        arrivalRooms: 85,
                        complimentRooms: 1,
                        houseUse: 3,
                        individualConfirm: 12,
                        occupancyPercentage: 31.82,
                        roomRevenue: 4200,
                        averageRoomRate: 350.50,
                        departureRooms: 38,
                        oooRooms: 1,
                        pax: 48,
                        totalRoomInventory: 132
                    },
                    {
                        id: '3',
                        date: '2025-08-12T18:34:46.908+00:00',
                        roomsSold: 58,
                        day: 'Wednesday',
                        arrivalRooms: 95,
                        complimentRooms: 0,
                        houseUse: 2,
                        individualConfirm: 15,
                        occupancyPercentage: 43.94,
                        roomRevenue: 5800,
                        averageRoomRate: 420.25,
                        departureRooms: 52,
                        oooRooms: 0,
                        pax: 65,
                        totalRoomInventory: 132
                    },
                    {
                        id: '4',
                        date: '2025-08-11T18:34:46.908+00:00',
                        roomsSold: 35,
                        day: 'Tuesday',
                        arrivalRooms: 70,
                        complimentRooms: 2,
                        houseUse: 1,
                        individualConfirm: 10,
                        occupancyPercentage: 26.52,
                        roomRevenue: 3150,
                        averageRoomRate: 380.75,
                        departureRooms: 30,
                        oooRooms: 2,
                        pax: 38,
                        totalRoomInventory: 132
                    },
                    {
                        id: '5',
                        date: '2025-08-10T18:34:46.908+00:00',
                        roomsSold: 72,
                        day: 'Monday',
                        arrivalRooms: 110,
                        complimentRooms: 3,
                        houseUse: 4,
                        individualConfirm: 20,
                        occupancyPercentage: 54.55,
                        roomRevenue: 7200,
                        averageRoomRate: 450.00,
                        departureRooms: 65,
                        oooRooms: 2,
                        pax: 85,
                        totalRoomInventory: 132
                    }
                ]);
            } catch (error) {
                console.error('Error fetching records:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchRecords();
    }, [hotelId]);

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
                            <th>Compliment</th>
                            <th>House Use</th>
                            <th>Individual Confirm</th>
                        </tr>
                    </thead>
                    <tbody>
                        {records.map(record => (
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
