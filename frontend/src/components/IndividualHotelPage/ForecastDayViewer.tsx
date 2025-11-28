import React, { useState, useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { fetchForecastAvailableDates, fetchSingleDayForecast } from '../../redux/services';

interface ForecastDayViewerProps {
    hotelId: string;
}

const ForecastDayViewer: React.FC<ForecastDayViewerProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const { availableDates, minDate, maxDate, loading } = useSelector(
        (state: RootState) => state.forecast
    );

    const [selectedDate, setSelectedDate] = useState<string>('');
    const [showCalendar, setShowCalendar] = useState(false);
    const [dayForecast, setDayForecast] = useState<any>(null);
    const [isLoadingForecast, setIsLoadingForecast] = useState(false);
    const [forecastError, setForecastError] = useState<string | null>(null);
    const [currentMonth, setCurrentMonth] = useState<Date>(new Date());
    const calendarRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (hotelId) {
            fetchAvailableForecastDates();
        }
    }, [hotelId]);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (calendarRef.current && !calendarRef.current.contains(event.target as Node)) {
                setShowCalendar(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    const fetchAvailableForecastDates = async () => {
        try {
            const response = await dispatch(fetchForecastAvailableDates(hotelId) as any);
            if (response?.minDate) {
                setSelectedDate(response.minDate);
                fetchDayForecast(response.minDate);
            }
        } catch (err: any) {
            console.error('Failed to fetch available dates:', err);
        }
    };

    const fetchDayForecast = async (date: string) => {
        try {
            setIsLoadingForecast(true);
            setForecastError(null);
            const forecast = await dispatch(fetchSingleDayForecast(hotelId, date) as any);
            if (forecast) {
                setDayForecast(forecast.forecast || forecast);
            }
        } catch (err: any) {
            setForecastError(err.response?.data?.message || 'Failed to fetch forecast data');
            setDayForecast(null);
        } finally {
            setIsLoadingForecast(false);
        }
    };

    const handleDateSelect = (date: string) => {
        setSelectedDate(date);
        fetchDayForecast(date);
        setShowCalendar(false);
    };

    const isDateAvailable = (date: string): boolean => {
        return availableDates.includes(date);
    };

    const getDayClass = (date: string, isCurrentMonth: boolean): string => {
        const isAvailable = isDateAvailable(date);
        const isSelected = date === selectedDate;

        let classes = "h-8 w-8 rounded-full flex items-center justify-center text-sm transition-colors duration-200 ";

        if (!isCurrentMonth) {
            classes += "text-gray-300 cursor-default";
        } else if (isSelected) {
            classes += "bg-indigo-600 text-white font-bold shadow-md";
        } else if (isAvailable) {
            classes += "text-gray-700 hover:bg-indigo-50 font-medium cursor-pointer";
        } else {
            classes += "text-gray-300 cursor-not-allowed";
        }

        return classes;
    };

    const renderCalendar = () => {
        if (!minDate || !maxDate) return null;

        const monthDays = [];
        const firstDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
        const lastDay = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0);

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
            const date = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), i);
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

        const monthYear = currentMonth.toLocaleString('en-US', { month: 'long', year: 'numeric' });

        return (
            <div className="absolute top-full left-0 mt-2 bg-white rounded-xl shadow-xl border border-gray-100 p-4 z-50 w-72">
                <div className="flex items-center justify-between mb-4">
                    <button
                        className="p-1 hover:bg-gray-100 rounded-full text-gray-600 transition-colors"
                        onClick={() => {
                            const newMonth = new Date(currentMonth);
                            newMonth.setMonth(newMonth.getMonth() - 1);
                            setCurrentMonth(newMonth);
                        }}
                        disabled={new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1) <= new Date(minDate)}
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                    </button>
                    <h4 className="text-sm font-bold text-gray-800">{monthYear}</h4>
                    <button
                        className="p-1 hover:bg-gray-100 rounded-full text-gray-600 transition-colors"
                        onClick={() => {
                            const newMonth = new Date(currentMonth);
                            newMonth.setMonth(newMonth.getMonth() + 1);
                            setCurrentMonth(newMonth);
                        }}
                        disabled={new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0) >= new Date(maxDate)}
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </button>
                </div>
                <div className="grid grid-cols-7 gap-1 mb-2">
                    {['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].map(day => (
                        <div key={day} className="text-center text-xs font-medium text-gray-400 py-1">
                            {day}
                        </div>
                    ))}
                </div>
                <div className="grid grid-cols-7 gap-1">
                    {monthDays.map((day, dayIdx) => (
                        <button
                            key={dayIdx}
                            className={getDayClass(day.date, day.isCurrentMonth)}
                            onClick={() => handleDateSelect(day.date)}
                            disabled={!day.isCurrentMonth || !isDateAvailable(day.date)}
                        >
                            {new Date(day.date).getDate()}
                        </button>
                    ))}
                </div>
            </div>
        );
    };

    return (
        <div className="w-full">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
                    <h3 className="text-lg font-bold text-gray-900 flex items-center">
                        <span className="mr-2">📅</span> Daily Forecast Details
                    </h3>
                    
                    <div className="relative" ref={calendarRef}>
                        <button
                            onClick={() => setShowCalendar(!showCalendar)}
                            className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 shadow-sm"
                        >
                            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                            <span>{selectedDate ? new Date(selectedDate).toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }) : 'Select Date'}</span>
                            <svg className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${showCalendar ? 'transform rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                        </button>
                        {showCalendar && renderCalendar()}
                    </div>
                </div>

                <div className="min-h-[200px]">
                    {loading ? (
                        <div className="flex items-center justify-center h-48 text-gray-500">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mr-3"></div>
                            Loading available dates...
                        </div>
                    ) : (
                        <>
                            {isLoadingForecast ? (
                                <div className="flex items-center justify-center h-48 text-gray-500">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mr-3"></div>
                                    Loading forecast details...
                                </div>
                            ) : forecastError ? (
                                <div className="flex items-center justify-center h-48 text-red-500 bg-red-50 rounded-lg border border-red-100">
                                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    {forecastError}
                                </div>
                            ) : dayForecast ? (
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                                    <div className="bg-gradient-to-br from-indigo-50 to-white p-4 rounded-xl border border-indigo-100 shadow-sm hover:shadow-md transition-shadow duration-200">
                                        <div className="text-xs font-medium text-indigo-600 uppercase tracking-wider mb-1">Revenue</div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            ₹{dayForecast.revenue?.toLocaleString('en-IN') || '0'}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-emerald-50 to-white p-4 rounded-xl border border-emerald-100 shadow-sm hover:shadow-md transition-shadow duration-200">
                                        <div className="text-xs font-medium text-emerald-600 uppercase tracking-wider mb-1">Rooms Sold</div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            {dayForecast.roomSold || 0}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-blue-50 to-white p-4 rounded-xl border border-blue-100 shadow-sm hover:shadow-md transition-shadow duration-200">
                                        <div className="text-xs font-medium text-blue-600 uppercase tracking-wider mb-1">Arrival Rooms</div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            {dayForecast.arrivalRoom || 0}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-amber-50 to-white p-4 rounded-xl border border-amber-100 shadow-sm hover:shadow-md transition-shadow duration-200">
                                        <div className="text-xs font-medium text-amber-600 uppercase tracking-wider mb-1">Departure Rooms</div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            {dayForecast.departureRoom || 0}
                                        </div>
                                    </div>
                                    <div className="bg-gradient-to-br from-rose-50 to-white p-4 rounded-xl border border-rose-100 shadow-sm hover:shadow-md transition-shadow duration-200">
                                        <div className="text-xs font-medium text-rose-600 uppercase tracking-wider mb-1">OOO Rooms</div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            {dayForecast.oooRoom || 0}
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-48 text-gray-400 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                                    <svg className="w-10 h-10 mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                    </svg>
                                    <span>No forecast data available. Please select a date.</span>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ForecastDayViewer;

