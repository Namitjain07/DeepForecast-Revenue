import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { fetchForecastAvailableDates, downloadForecastCSV } from '../../redux/services/api';

interface DownloadForcastCSVProps {
    hotelId: string;
}

const DownloadForcastCSV: React.FC<DownloadForcastCSVProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { loading, error, availableDates, minDate, maxDate } = useSelector(
        (state: RootState) => state.forecast
    );

    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [isDownloading, setIsDownloading] = useState(false);
    const [downloadError, setDownloadError] = useState<string | null>(null);
    const [showStartCalendar, setShowStartCalendar] = useState(false);
    const [showEndCalendar, setShowEndCalendar] = useState(false);

    useEffect(() => {
        if (hotelId) {
            fetchAvailableForecastDates();
        }
    }, [hotelId]);

    const fetchAvailableForecastDates = async () => {
        try {
            const response = await dispatch(fetchForecastAvailableDates(hotelId) as any);
            if (response.minDate && response.maxDate) {
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
            await dispatch(downloadForecastCSV(hotelId, startDate, endDate) as any);
        } catch (err: any) {
            setDownloadError(err.response?.data?.message || 'Failed to download CSV');
        } finally {
            setIsDownloading(false);
        }
    };

    const isDateAvailable = (date: string): boolean => {
        return availableDates.includes(date);
    };

    const getDayClass = (date: string): string => {
        const isAvailable = isDateAvailable(date);
        const isStart = date === startDate;
        const isEnd = date === endDate;
        const isInRange = date >= startDate && date <= endDate && startDate && endDate;

        if (isStart) return 'bg-indigo-600 text-white hover:bg-indigo-700';
        if (isEnd) return 'bg-indigo-600 text-white hover:bg-indigo-700';
        if (isInRange) return 'bg-indigo-100 text-indigo-900';
        if (isAvailable) return 'text-gray-900 hover:bg-gray-100 font-medium';
        return 'text-gray-300 cursor-not-allowed';
    };

    const renderCalendar = (_selectedDate: string, isStartDate: boolean) => {
        if (!minDate || !maxDate) return null;

        const months = [];
        const start = new Date(minDate);
        const end = new Date(maxDate);

        const monthStart = new Date(start.getFullYear(), start.getMonth(), 1);
        const monthEnd = new Date(end.getFullYear(), end.getMonth(), 1);

        while (monthStart <= monthEnd) {
            const monthYear = monthStart.toLocaleString('en-US', { month: 'long', year: 'numeric' });
            months.push({
                monthYear,
                month: monthStart.getMonth(),
                year: monthStart.getFullYear()
            });
            monthStart.setMonth(monthStart.getMonth() + 1);
        }

        return (
            <div className="absolute z-10 mt-2 bg-white rounded-xl shadow-xl border border-gray-200 p-4 w-80 max-h-96 overflow-y-auto">
                {months.map((monthData, idx) => {
                    const monthDays = [];
                    const firstDay = new Date(monthData.year, monthData.month, 1);
                    const lastDay = new Date(monthData.year, monthData.month + 1, 0);

                    const startDay = firstDay.getDay();
                    for (let i = startDay - 1; i >= 0; i--) {
                        const prevDate = new Date(firstDay);
                        prevDate.setDate(prevDate.getDate() - (i + 1));
                        monthDays.push({
                            date: prevDate.toISOString().split('T')[0],
                            isCurrentMonth: false
                        });
                    }

                    for (let i = 1; i <= lastDay.getDate(); i++) {
                        const date = new Date(monthData.year, monthData.month, i);
                        monthDays.push({
                            date: date.toISOString().split('T')[0],
                            isCurrentMonth: true
                        });
                    }

                    const remainingDays = 42 - monthDays.length;
                    for (let i = 1; i <= remainingDays; i++) {
                        const nextDate = new Date(lastDay);
                        nextDate.setDate(nextDate.getDate() + i);
                        monthDays.push({
                            date: nextDate.toISOString().split('T')[0],
                            isCurrentMonth: false
                        });
                    }

                    return (
                        <div key={idx} className="mb-6 last:mb-0">
                            <h4 className="text-sm font-bold text-gray-900 mb-3 text-center">{monthData.monthYear}</h4>
                            <div className="grid grid-cols-7 gap-1 mb-2">
                                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                                    <div key={day} className="text-xs font-medium text-gray-400 text-center">{day}</div>
                                ))}
                            </div>
                            <div className="grid grid-cols-7 gap-1">
                                {monthDays.map((day, dayIdx) => (
                                    <button
                                        key={dayIdx}
                                        className={`
                                            h-8 w-8 rounded-full text-xs flex items-center justify-center transition-colors duration-150
                                            ${day.isCurrentMonth ? '' : 'invisible'}
                                            ${getDayClass(day.date)}
                                        `}
                                        onClick={() => {
                                            if (isDateAvailable(day.date)) {
                                                if (isStartDate) {
                                                    setStartDate(day.date);
                                                    setShowStartCalendar(false);
                                                } else {
                                                    setEndDate(day.date);
                                                    setShowEndCalendar(false);
                                                }
                                            }
                                        }}
                                        disabled={!day.isCurrentMonth || !isDateAvailable(day.date)}
                                    >
                                        {new Date(day.date).getDate()}
                                    </button>
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>
        );
    };

    return (
        <div className="w-full">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <h2 className="text-lg font-bold text-gray-900 mb-6 flex items-center">
                    <span className="mr-2">📈</span> Download Forecast
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
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="relative">
                                    <div className="flex flex-col">
                                        <label className="text-sm font-medium text-gray-700 mb-2">Start Date</label>
                                        <button
                                            className="w-full px-4 py-2 text-left border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all duration-200 bg-white text-gray-900"
                                            onClick={() => setShowStartCalendar(!showStartCalendar)}
                                        >
                                            {startDate || 'Select Start Date'}
                                        </button>
                                    </div>
                                    {showStartCalendar && (
                                        <>
                                            <div className="fixed inset-0 z-0" onClick={() => setShowStartCalendar(false)}></div>
                                            {renderCalendar(startDate, true)}
                                        </>
                                    )}
                                </div>

                                <div className="relative">
                                    <div className="flex flex-col">
                                        <label className="text-sm font-medium text-gray-700 mb-2">End Date</label>
                                        <button
                                            className="w-full px-4 py-2 text-left border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all duration-200 bg-white text-gray-900"
                                            onClick={() => setShowEndCalendar(!showEndCalendar)}
                                        >
                                            {endDate || 'Select End Date'}
                                        </button>
                                    </div>
                                    {showEndCalendar && (
                                        <>
                                            <div className="fixed inset-0 z-0" onClick={() => setShowEndCalendar(false)}></div>
                                            {renderCalendar(endDate, false)}
                                        </>
                                    )}
                                </div>
                            </div>

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

                            <div className="bg-gray-50 rounded-lg p-4 border border-gray-100">
                                <p className="text-sm text-gray-600 mb-2">
                                    Download forecast data in CSV format for the selected date range.
                                    The file will include all forecast data including revenue, rooms sold, and occupancy predictions.
                                </p>
                                {minDate && maxDate && (
                                    <p className="text-xs text-gray-500 font-medium">
                                        Available data range: <span className="text-indigo-600">{minDate}</span> to <span className="text-indigo-600">{maxDate}</span>
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

export default DownloadForcastCSV;

