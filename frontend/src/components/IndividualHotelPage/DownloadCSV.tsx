import React, { useState } from 'react';
import '../../stylesheet/ui/component-ui-download-csv.css';

interface DownloadCSVProps {
    hotelId: string;
}

const DownloadCSV: React.FC<DownloadCSVProps> = ({ hotelId }) => {
    const [startDate, setStartDate] = useState('2025-08-01');
    const [endDate, setEndDate] = useState('2025-08-31');
    const [isDownloading, setIsDownloading] = useState(false);

    const handleDownload = () => {
        if (!startDate || !endDate) {
            alert('Please select both start and end dates');
            return;
        }

        if (new Date(startDate) > new Date(endDate)) {
            alert('Start date cannot be after end date');
            return;
        }

        setIsDownloading(true);

        // Simulate download
        setTimeout(() => {
            console.log(`Downloading CSV for hotel ${hotelId} from ${startDate} to ${endDate}`);
            const element = document.createElement('a');
            const file = new Blob([`Date,Revenue,Rooms Sold\n${startDate},10000,50`], {type: 'text/csv'});
            element.href = URL.createObjectURL(file);
            element.download = `records_${startDate}_to_${endDate}.csv`;
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
            setIsDownloading(false);
        }, 500);
    };

    return (
        <div className="component-ui-download-csv-container">
            <h2 className="component-ui-download-csv-title">Download Records</h2>
            <div className="component-ui-download-csv-content">
                <div className="component-ui-download-csv-form">
                    <div className="component-ui-download-csv-form-group">
                        <label htmlFor="startDate" className="component-ui-download-csv-label">
                            Start Date
                        </label>
                        <input
                            type="date"
                            id="startDate"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="component-ui-download-csv-input"
                        />
                    </div>

                    <div className="component-ui-download-csv-form-group">
                        <label htmlFor="endDate" className="component-ui-download-csv-label">
                            End Date
                        </label>
                        <input
                            type="date"
                            id="endDate"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="component-ui-download-csv-input"
                        />
                    </div>

                    <button
                        onClick={handleDownload}
                        disabled={isDownloading}
                        className="component-ui-download-csv-button"
                    >
                        {isDownloading ? 'Downloading...' : 'Download CSV'}
                    </button>
                </div>

                <div className="component-ui-download-csv-info">
                    <p className="component-ui-download-csv-info-text">
                        Download hotel records in CSV format for the selected date range.
                        The file will include all transaction data including revenue, rooms sold, and occupancy information.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default DownloadCSV;
