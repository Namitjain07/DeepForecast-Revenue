import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { downloadRecordsCSV, fetchAvailableDates } from '../../redux/services/api';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';

interface DownloadCSVProps {
    hotelId: string;
}

const DownloadRecordCSV: React.FC<DownloadCSVProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { loading, error } = useSelector((state: RootState) => state.records);

    const [startDate, setStartDate] = useState<string>('');
    const [endDate, setEndDate] = useState<string>('');
    const [isDownloading, setIsDownloading] = useState(false);
    const [downloadError, setDownloadError] = useState<string | null>(null);
    const [, setAvailableDates] = useState<string[]>([]);
    const [minDate, setMinDate] = useState<string>('');
    const [maxDate, setMaxDate] = useState<string>('');

    useEffect(() => {
        if (hotelId) {
            fetchAvailableDatesForHotel();
        }
    }, [hotelId]);

    const fetchAvailableDatesForHotel = async () => {
        try {
            const response = await dispatch(fetchAvailableDates(hotelId) as any);
            if (response.dates) {
                setAvailableDates(response.dates);
                setMinDate(response.minDate);
                setMaxDate(response.maxDate);
                setStartDate(response.minDate);
                setEndDate(response.maxDate);
            }
        } catch (err: any) {
            console.error('Failed to fetch available dates:', err);
        }
    };

    const handleDownload = async () => {
        if (!startDate || !endDate) {
            setDownloadError('Please select both start and end dates');
            return;
        }

        if (new Date(startDate) > new Date(endDate)) {
            setDownloadError('Start date cannot be after end date');
            return;
        }

        setDownloadError(null);
        setIsDownloading(true);

        try {
            await dispatch(downloadRecordsCSV(hotelId, startDate, endDate) as any);
        } catch (err: any) {
            setDownloadError(err.response?.data?.message || 'Failed to download CSV');
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <div className="w-full">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <h2 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
                    <span className="mr-2">📥</span> Download Records
                </h2>

                <div className="space-y-6">
                    {loading ? (
                        <div className="flex items-center justify-center py-8 text-gray-500">
                            <svg className="animate-spin h-5 w-5 mr-3 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Loading available dates...
                        </div>
                    ) : (
                        <>
                            {/* Date Picker Section */}
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="flex flex-col">
                                    <label className="text-sm font-medium text-gray-700 mb-2">Start Date</label>
                                    <div className="relative">
                                        <DatePicker
                                            selected={startDate ? new Date(startDate) : null}
                                            onChange={(date: Date | null) =>
                                                setStartDate(date ? date.toISOString().split('T')[0] : '')
                                            }
                                            minDate={minDate ? new Date(minDate) : undefined}
                                            maxDate={maxDate ? new Date(maxDate) : undefined}
                                            dateFormat="yyyy-MM-dd"
                                            placeholderText="Select start date"
                                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all duration-200"
                                            popperPlacement="bottom-start"
                                        />
                                    </div>
                                </div>

                                <div className="flex flex-col">
                                    <label className="text-sm font-medium text-gray-700 mb-2">End Date</label>
                                    <div className="relative">
                                        <DatePicker
                                            selected={endDate ? new Date(endDate) : null}
                                            onChange={(date: Date | null) =>
                                                setEndDate(date ? date.toISOString().split('T')[0] : '')
                                            }
                                            minDate={minDate ? new Date(minDate) : undefined}
                                            maxDate={maxDate ? new Date(maxDate) : undefined}
                                            dateFormat="yyyy-MM-dd"
                                            placeholderText="Select end date"
                                            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all duration-200"
                                            popperPlacement="bottom-start"
                                        />
                                    </div>
                                </div>
                            </div>

                            {/* Download Button */}
                            <div className="flex flex-col items-end space-y-3">
                                <button
                                    onClick={handleDownload}
                                    disabled={isDownloading || !startDate || !endDate}
                                    className={`
                                        inline-flex items-center px-6 py-2.5 rounded-lg text-sm font-medium text-white shadow-md transition-all duration-200
                                        ${isDownloading || !startDate || !endDate
                                            ? 'bg-gray-300 cursor-not-allowed'
                                            : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 hover:shadow-lg hover:-translate-y-0.5'
                                        }
                                    `}
                                >
                                    {isDownloading ? (
                                        <>
                                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            Downloading...
                                        </>
                                    ) : (
                                        <>
                                            <span className="mr-2">⬇️</span> Download CSV
                                        </>
                                    )}
                                </button>

                                {(downloadError || error) && (
                                    <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg border border-red-100">
                                        {downloadError || error}
                                    </div>
                                )}
                            </div>

                            {/* Info Section */}
                            <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                                <p className="text-sm text-gray-600 mb-2">
                                    Download hotel records in CSV format for the selected date range.
                                    The file includes all data such as revenue, rooms sold, and occupancy information.
                                </p>

                                {minDate && maxDate && (
                                    <p className="text-xs text-gray-500 font-medium">
                                        Available data range: <span className="text-indigo-600">{minDate}</span> to{' '}
                                        <span className="text-indigo-600">{maxDate}</span>
                                    </p>
                                )}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DownloadRecordCSV;
