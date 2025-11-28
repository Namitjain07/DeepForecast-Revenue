import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type {RootState, AppDispatch} from '../../redux/store';
import { fetchRecentRecords } from '../../redux/services/api';

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
        });
    };

    if (loading) {
        return (
            <div className="flex justify-center items-center h-32 text-gray-500">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-indigo-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Loading records...
            </div>
        );
    }

    if (error) {
        return <div className="p-4 text-red-600 bg-red-50 rounded-lg border border-red-100">Error: {error}</div>;
    }

    if (!recentRecords || recentRecords.length === 0) {
        return <div className="p-8 text-center text-gray-500 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">No recent records found</div>;
    }

    return (
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center mb-6">
                <div className="bg-indigo-100 p-2 rounded-lg mr-3">
                    <span className="text-2xl">📋</span>
                </div>
                <h2 className="text-xl font-bold text-gray-900">Recent Records</h2>
            </div>
            <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                        <tr>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Date</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Day</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Rooms Sold</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Occupancy %</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Revenue</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">ADR</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Arr/Dep</th>
                            <th scope="col" className="px-6 py-3 text-left text-xs font-bold text-gray-600 uppercase tracking-wider whitespace-nowrap">Inventory</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {recentRecords.map((record) => (
                            <tr key={record.id} className="hover:bg-indigo-50/50 transition-colors duration-150">
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{formatDate(record.date)}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{record.day}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">{record.roomsSold}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    <span className={`px-2 py-1 inline-flex text-xs leading-5 font-bold rounded-full shadow-sm ${
                                        record.occupancyPercentage >= 80 ? 'bg-green-100 text-green-800 border border-green-200' : 
                                        record.occupancyPercentage >= 50 ? 'bg-yellow-100 text-yellow-800 border border-yellow-200' : 
                                        'bg-red-100 text-red-800 border border-red-200'
                                    }`}>
                                        {record.occupancyPercentage.toFixed(1)}%
                                    </span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-indigo-600">₹{record.roomRevenue.toLocaleString('en-IN')}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">₹{record.averageRoomRate.toFixed(0)}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                    <span className="text-green-600 font-medium">{record.arrivalRooms}</span>
                                    <span className="mx-1 text-gray-300">/</span>
                                    <span className="text-red-600 font-medium">{record.departureRooms}</span>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{record.totalRoomInventory}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default RecentRecords;

