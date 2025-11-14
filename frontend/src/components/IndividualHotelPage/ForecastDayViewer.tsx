import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { fetchForecastAvailableDates, fetchSingleDayForecast } from '../../redux/services';
import '../../stylesheet/ui/component-ui-forecast-day-viewer.css';

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

    useEffect(() => {
        if (hotelId) {
            fetchAvailableForecastDates();
        }
    }, [hotelId]);

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
            setForecastError(err.message || 'Failed to fetch forecast data');
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

    const getDayClass = (date: string): string => {
        const isAvailable = isDateAvailable(date);
        const isSelected = date === selectedDate;

        if (isSelected) return 'ui-component-selected';
        if (isAvailable) return 'ui-component-available';
        return '';
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
            <div className="ui-component-forecast-day-calendar">
                <div className="ui-component-forecast-day-month-navigation">
                    <button
                        className="ui-component-forecast-day-nav-btn"
                        onClick={() => {
                            const newMonth = new Date(currentMonth);
                            newMonth.setMonth(newMonth.getMonth() - 1);
                            setCurrentMonth(newMonth);
                        }}
                        disabled={new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1) <= new Date(minDate)}
                    >
                        ◀ Prev
                    </button>
                    <h4 className="ui-component-forecast-day-month-header">{monthYear}</h4>
                    <button
                        className="ui-component-forecast-day-nav-btn"
                        onClick={() => {
                            const newMonth = new Date(currentMonth);
                            newMonth.setMonth(newMonth.getMonth() + 1);
                            setCurrentMonth(newMonth);
                        }}
                        disabled={new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0) >= new Date(maxDate)}
                    >
                        Next ▶
                    </button>
                </div>
                <div className="ui-component-forecast-day-month">
                    <div className="ui-component-forecast-day-weekdays">
                        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                            <div key={day} className="ui-component-forecast-day-weekday">{day}</div>
                        ))}
                    </div>
                    <div className="ui-component-forecast-day-days">
                        {monthDays.map((day, dayIdx) => (
                            <button
                                key={dayIdx}
                                className={`ui-component-forecast-day-day ${day.isCurrentMonth ? '' : 'ui-component-forecast-day-other-month'} ${getDayClass(day.date)}`}
                                onClick={() => handleDateSelect(day.date)}
                                disabled={!day.isCurrentMonth || !isDateAvailable(day.date)}
                            >
                                {new Date(day.date).getDate()}
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="ui-component-forecast-day-viewer-container">
            <h2 className="ui-component-forecast-day-viewer-title">Daily Forecast Details</h2>
            <div className="ui-component-forecast-day-viewer-content">
                {loading ? (
                    <div className="ui-component-forecast-day-viewer-loading">Loading available dates...</div>
                ) : (
                    <div className="ui-component-forecast-day-viewer-main">
                        {/* Calendar Section */}
                        <div className="ui-component-forecast-day-calendar-section">
                            <div className="ui-component-forecast-day-calendar-header">
                                <label>Select Date</label>
                                <button
                                    className="ui-component-forecast-day-toggle"
                                    onClick={() => setShowCalendar(!showCalendar)}
                                >
                                    📅 {selectedDate || 'Select Date'}
                                </button>
                            </div>
                            {showCalendar && (
                                <div className="ui-component-forecast-day-calendar-dropdown">
                                    {renderCalendar()}
                                </div>
                            )}
                        </div>

                        {/* Forecast Details Section */}
                        <div className="ui-component-forecast-day-details-section">
                            {isLoadingForecast ? (
                                <div className="ui-component-forecast-day-viewer-loading">
                                    Loading forecast details...
                                </div>
                            ) : forecastError ? (
                                <div className="ui-component-forecast-day-viewer-error">
                                    ⚠️ {forecastError}
                                </div>
                            ) : dayForecast ? (
                                <div className="ui-component-forecast-day-details-grid">
                                    <div className="ui-component-forecast-day-detail-card">
                                        <span className="ui-component-forecast-day-detail-label">Revenue</span>
                                        <span className="ui-component-forecast-day-detail-value">
                                            ₹{dayForecast.revenue?.toLocaleString('en-IN') || '0'}
                                        </span>
                                    </div>
                                    <div className="ui-component-forecast-day-detail-card">
                                        <span className="ui-component-forecast-day-detail-label">Rooms Sold</span>
                                        <span className="ui-component-forecast-day-detail-value">
                                            {dayForecast.roomSold || 0}
                                        </span>
                                    </div>
                                    <div className="ui-component-forecast-day-detail-card">
                                        <span className="ui-component-forecast-day-detail-label">Arrival Rooms</span>
                                        <span className="ui-component-forecast-day-detail-value">
                                            {dayForecast.arrivalRoom || 0}
                                        </span>
                                    </div>
                                    <div className="ui-component-forecast-day-detail-card">
                                        <span className="ui-component-forecast-day-detail-label">Departure Rooms</span>
                                        <span className="ui-component-forecast-day-detail-value">
                                            {dayForecast.departureRoom || 0}
                                        </span>
                                    </div>
                                    <div className="ui-component-forecast-day-detail-card">
                                        <span className="ui-component-forecast-day-detail-label">OOO Rooms</span>
                                        <span className="ui-component-forecast-day-detail-value">
                                            {dayForecast.oooRoom || 0}
                                        </span>
                                    </div>


                                </div>
                            ) : (
                                <div className="ui-component-forecast-day-viewer-empty">
                                    No forecast data available. Please select a date.
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default ForecastDayViewer;

