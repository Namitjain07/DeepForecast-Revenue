import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { fetchForecastAvailableDates, downloadForecastCSV } from '../../redux/services/api';
import '../../stylesheet/ui/component-ui-download-forcast-csv.css';

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
            setDownloadError(err.message || 'Failed to download CSV');
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

        if (isStart) return 'ui-component-selected ui-component-start';
        if (isEnd) return 'ui-component-selected ui-component-end';
        if (isInRange) return 'ui-component-in-range';
        if (isAvailable) return 'ui-component-available';
        return '';
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
            <div className="ui-component-calendar">
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
                        <div key={idx} className="ui-component-month">
                            <h4 className="ui-component-month-header">{monthData.monthYear}</h4>
                            <div className="ui-component-weekdays">
                                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                                    <div key={day} className="ui-component-weekday">{day}</div>
                                ))}
                            </div>
                            <div className="ui-component-days">
                                {monthDays.map((day, dayIdx) => (
                                    <button
                                        key={dayIdx}
                                        className={`ui-component-day ${day.isCurrentMonth ? '' : 'ui-component-other-month'} ${getDayClass(day.date)}`}
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
        <div className="ui-component-download-forcast-container">
            <h2 className="ui-component-download-forcast-title">Download Forecast</h2>
            <div className="ui-component-download-forcast-content">
                {loading ? (
                    <div className="ui-component-download-forcast-loading">Loading available dates...</div>
                ) : (
                    <>
                        <div className="ui-component-download-forcast-calendars">
                            <div className="ui-component-calendar-wrapper">
                                <div className="ui-component-calendar-label">
                                    <label>Start Date</label>
                                    <button
                                        className="ui-component-date-toggle"
                                        onClick={() => setShowStartCalendar(!showStartCalendar)}
                                    >
                                        {startDate || 'Select Start Date'}
                                    </button>
                                </div>
                                {showStartCalendar && (
                                    <div className="ui-component-calendar-dropdown">
                                        {renderCalendar(startDate, true)}
                                    </div>
                                )}
                            </div>

                            <div className="ui-component-calendar-wrapper">
                                <div className="ui-component-calendar-label">
                                    <label>End Date</label>
                                    <button
                                        className="ui-component-date-toggle"
                                        onClick={() => setShowEndCalendar(!showEndCalendar)}
                                    >
                                        {endDate || 'Select End Date'}
                                    </button>
                                </div>
                                {showEndCalendar && (
                                    <div className="ui-component-calendar-dropdown">
                                        {renderCalendar(endDate, false)}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="ui-component-download-forcast-form">
                            <button
                                onClick={handleDownload}
                                disabled={isDownloading || !startDate || !endDate}
                                className="ui-component-download-forcast-button"
                            >
                                {isDownloading ? 'Downloading...' : 'Download CSV'}
                            </button>

                            {(downloadError || error) && (
                                <div className="ui-component-download-forcast-error">
                                    {downloadError || error}
                                </div>
                            )}
                        </div>

                        <div className="ui-component-download-forcast-info">
                            <p className="ui-component-download-forcast-info-text">
                                Download forecast data in CSV format for the selected date range.
                                The file will include all forecast data including revenue, rooms sold, and occupancy predictions.
                            </p>
                            {minDate && maxDate && (
                                <p className="ui-component-download-forcast-info-dates">
                                    Available data range: {minDate} to {maxDate}
                                </p>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default DownloadForcastCSV;

