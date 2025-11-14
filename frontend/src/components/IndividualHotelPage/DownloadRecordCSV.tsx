import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { AppDispatch, RootState } from '../../redux/store';
import { downloadRecordsCSV, fetchAvailableDates } from '../../redux/services/api';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import '../../stylesheet/ui/component-ui-download-csv.css';

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
            setDownloadError(err.message || 'Failed to download CSV');
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <div className="component-ui-download-csv-container">
            <h2 className="component-ui-download-csv-title">Download Records</h2>

            <div className="component-ui-download-csv-content">
                {loading ? (
                    <div className="component-ui-download-csv-loading">Loading available dates...</div>
                ) : (
                    <>
                        {/* Date Picker Section */}
                        <div className="component-ui-download-csv-datepickers">
                            <div className="component-ui-download-csv-field">
                                <label>Start Date:</label>
                                <DatePicker
                                    selected={startDate ? new Date(startDate) : null}
                                    onChange={(date: Date | null) =>
                                        setStartDate(date ? date.toISOString().split('T')[0] : '')
                                    }
                                    minDate={minDate ? new Date(minDate) : undefined}
                                    maxDate={maxDate ? new Date(maxDate) : undefined}
                                    dateFormat="yyyy-MM-dd"
                                    placeholderText="Select start date"
                                    className="component-ui-download-csv-input"
                                    popperPlacement="bottom-start"
                                />
                            </div>

                            <div className="component-ui-download-csv-field">
                                <label>End Date:</label>
                                <DatePicker
                                    selected={endDate ? new Date(endDate) : null}
                                    onChange={(date: Date | null) =>
                                        setEndDate(date ? date.toISOString().split('T')[0] : '')
                                    }
                                    minDate={minDate ? new Date(minDate) : undefined}
                                    maxDate={maxDate ? new Date(maxDate) : undefined}
                                    dateFormat="yyyy-MM-dd"
                                    placeholderText="Select end date"
                                    className="component-ui-download-csv-input"
                                    popperPlacement="bottom-start"
                                />
                            </div>
                        </div>

                        {/* Download Button */}
                        <div className="component-ui-download-csv-form">
                            <button
                                onClick={handleDownload}
                                disabled={isDownloading || !startDate || !endDate}
                                className="component-ui-download-csv-button"
                            >
                                {isDownloading ? 'Downloading...' : 'Download CSV'}
                            </button>

                            {(downloadError || error) && (
                                <div className="component-ui-download-csv-error">
                                    {downloadError || error}
                                </div>
                            )}
                        </div>

                        {/* Info Section */}
                        <div className="component-ui-download-csv-info">
                            <p className="component-ui-download-csv-info-text">
                                Download hotel records in CSV format for the selected date range.
                                The file includes all data such as revenue, rooms sold, and occupancy information.
                            </p>

                            {minDate && maxDate && (
                                <p className="component-ui-download-csv-info-dates">
                                    Available data range: <strong>{minDate}</strong> to{' '}
                                    <strong>{maxDate}</strong>
                                </p>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};

export default DownloadRecordCSV;
