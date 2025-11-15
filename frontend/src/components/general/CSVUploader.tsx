import React, { useState, useRef,} from 'react';
import * as XLSX from 'xlsx';
import { useDispatch } from 'react-redux';
import type { AppDispatch } from '../../redux/store';
import { addRecordsToHotel } from '../../redux/services/recordsApi';
import '../../stylesheet/ui/component-ui-csv-uploader.css';

interface CSVUploaderProps {
    hotelId: string;
    onSuccess?: () => void;
}

const CSVUploader: React.FC<CSVUploaderProps> = ({ hotelId, onSuccess }) => {
    const dispatch = useDispatch<AppDispatch>();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const [uploadProgress, setUploadProgress] = useState(0);
    const requiredColumns = [
        'Date',
        'Day',
        'Rooms Sold',
        'Arrival Rooms',
        'Departure Rooms',
        'OOO Rooms',
        'Occupancy %',
        'Room Revenue',
        'ARR',
        'PAX',
        'Compliment Rooms',
        'House Use',
        'Individual Confirm',
        'Total Room Inventory'
    ];


    const parseXLSX = async (file: File): Promise<any[]> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                try {
                    const data = e.target?.result;
                    const workbook = XLSX.read(data, { type: 'binary' });
                    const sheetName = workbook.SheetNames[0];
                    const worksheet = workbook.Sheets[sheetName];

                    // Convert to JSON
                    const jsonData = XLSX.utils.sheet_to_json(worksheet);

                    if (jsonData.length === 0) {
                        throw new Error('Excel file contains no data');
                    }

                    // Validate and convert
                    const records = convertToRecords(jsonData);
                    resolve(records);
                } catch (err: any) {
                    reject(new Error(`Failed to parse XLSX: ${err.message}`));
                }
            };

            reader.onerror = () => {
                reject(new Error('Failed to read XLSX file'));
            };

            reader.readAsBinaryString(file);
        });
    };

    const parseCSV = (csvText: string): any[] => {
        const lines = csvText.trim().split('\n');
        if (lines.length < 2) {
            throw new Error('File must have at least header row and one data row');
        }

        // Parse header
        const headers = lines[0].split(',').map(h => h.trim());

        // Validate headers
        const missingColumns = requiredColumns.filter(col => !headers.includes(col));
        if (missingColumns.length > 0) {
            throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
        }

        // Parse data rows
        const records: any[] = [];
        for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim() === '') continue;

            const values = lines[i].split(',').map(v => v.trim());
            if (values.length < requiredColumns.length) {
                throw new Error(`Row ${i + 1} has fewer columns than expected. Expected ${requiredColumns.length}, got ${values.length}`);
            }

            const record: any = {};
            requiredColumns.forEach((col, index) => {
                const value = values[index];
                if (!value) {
                    throw new Error(`Row ${i + 1}, Column "${col}": Value cannot be empty`);
                }
                record[col] = value;
            });

            records.push(record);
        }

        if (records.length === 0) {
            throw new Error('File contains no data rows');
        }

        return records;
    };

    const convertToRecords = (data: any[]): any[] => {
        const records: any[] = [];

        data.forEach((row: any, index: number) => {
            // Validate all required columns exist
            const missingColumns = requiredColumns.filter(col => !(col in row));
            if (missingColumns.length > 0) {
                throw new Error(`Row ${index + 1} missing columns: ${missingColumns.join(', ')}`);
            }

            const record: any = {};
            requiredColumns.forEach(col => {
                const value = row[col];

                if (value === null || value === undefined || value === '') {
                    throw new Error(`Row ${index + 1}, Column "${col}": Value cannot be empty`);
                }

                // Map CSV column names to database field names
                switch (col) {
                    case 'Date':
                        record.date = String(value);
                        break;
                    case 'Day':
                        record.day = String(value);
                        break;
                    case 'Rooms Sold':
                        record.roomsSold = parseInt(String(value));
                        if (isNaN(record.roomsSold)) {
                            throw new Error(`Row ${index + 1}, Column "Rooms Sold": Must be a valid number`);
                        }
                        break;
                    case 'Arrival Rooms':
                        record.arrivalRooms = parseInt(String(value));
                        if (isNaN(record.arrivalRooms)) {
                            throw new Error(`Row ${index + 1}, Column "Arrival Rooms": Must be a valid number`);
                        }
                        break;
                    case 'Departure Rooms':
                        record.departureRooms = parseInt(String(value));
                        if (isNaN(record.departureRooms)) {
                            throw new Error(`Row ${index + 1}, Column "Departure Rooms": Must be a valid number`);
                        }
                        break;
                    case 'Occupancy %':
                        record.occupancyPercentage = parseFloat(String(value));
                        if (isNaN(record.occupancyPercentage)) {
                            throw new Error(`Row ${index + 1}, Column "Occupancy %": Must be a valid number`);
                        }
                        break;
                    case 'Room Revenue':
                        record.roomRevenue = parseInt(String(value));
                        if (isNaN(record.roomRevenue)) {
                            throw new Error(`Row ${index + 1}, Column "Room Revenue": Must be a valid number`);
                        }
                        break;
                    case 'ARR':
                        record.averageRoomRate = parseFloat(String(value));
                        if (isNaN(record.averageRoomRate)) {
                            throw new Error(`Row ${index + 1}, Column "Avg Room Rate": Must be a valid number`);
                        }
                        break;
                    case 'PAX':
                        record.pax = parseInt(String(value));
                        if (isNaN(record.pax)) {
                            throw new Error(`Row ${index + 1}, Column "PAX": Must be a valid number`);
                        }
                        break;
                    case 'Total Room Inventory':
                        record.totalRoomInventory = parseInt(String(value));
                        if (isNaN(record.totalRoomInventory)) {
                            throw new Error(`Row ${index + 1}, Column "Total Inventory": Must be a valid number`);
                        }
                        break;
                    case 'OOO Rooms':
                        record.oooRooms = parseInt(String(value));
                        if (isNaN(record.oooRooms)) {
                            throw new Error(`Row ${index + 1}, Column "OOO Rooms": Must be a valid number`);
                        }
                        break;
                    case 'Compliment Rooms':
                        record.complimentRooms = parseInt(String(value));
                        if (isNaN(record.complimentRooms)) {
                            throw new Error(`Row ${index + 1}, Column "Compliment Rooms": Must be a valid number`);
                        }
                        break;
                    case 'House Use':
                        record.houseUse = parseInt(String(value));
                        if (isNaN(record.houseUse)) {
                            throw new Error(`Row ${index + 1}, Column "House Use": Must be a valid number`);
                        }
                        break;
                    case 'Individual Confirm':
                        record.individualConfirm = parseInt(String(value));
                        if (isNaN(record.individualConfirm)) {
                            throw new Error(`Row ${index + 1}, Column "Individual Confirm": Must be a valid number`);
                        }
                        break;
                }
            });

            records.push(record);
        });

        return records;
    };

    const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // Validate file type
        const isCSV = file.name.endsWith('.csv');
        const isXLSX = file.name.endsWith('.xlsx') || file.name.endsWith('.xls');

        if (!isCSV && !isXLSX) {
            setError('Please upload a CSV or XLSX file');
            return;
        }

        try {
            setIsProcessing(true);
            setError(null);
            setSuccess(null);
            setUploadProgress(0);

            let records: any[];

            if (isXLSX) {
                // Parse XLSX
                records = await parseXLSX(file);
                console.log(records)
                setUploadProgress(60);
            } else {
                // Parse CSV
                const fileContent = await file.text();
                setUploadProgress(30);
                records = parseCSV(fileContent);
                setUploadProgress(60);
            }

            // Send to backend via Redux
            const result = await dispatch(addRecordsToHotel({ hotelId, records }) as any);
            setUploadProgress(90);

            if (!result || result.error) {
                throw new Error(result?.message || 'Failed to upload records');
            }

            setUploadProgress(100);
            setSuccess(`✓ Successfully uploaded ${result.count || records.length} records`);

            // Reset file input
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }

            // Call callback after success
            setTimeout(() => {
                if (onSuccess) {
                    onSuccess();
                }
            }, 1500);
        } catch (err: any) {
            setError(`✕ ${err.response?.data?.message}`);
            console.error('File upload error:', err);
        } finally {
            setIsProcessing(false);
            setUploadProgress(0);
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    return (
        <div className="component-ui-csv-uploader-container">
            <div className="component-ui-csv-uploader-card">
                <h3 className="component-ui-csv-uploader-title">📊 Upload Records via CSV/XLSX</h3>
                <p className="component-ui-csv-uploader-description">
                    Upload hotel records from a CSV or Excel file. The file must contain all required columns.
                </p>

                {/* Error/Success Messages */}
                {error && (
                    <div className="component-ui-csv-uploader-error">
                        {error}
                    </div>
                )}
                {success && (
                    <div className="component-ui-csv-uploader-success">
                        {success}
                    </div>
                )}

                {/* Upload Area */}
                <div
                    className="component-ui-csv-uploader-zone"
                    onClick={handleClick}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.xlsx,.xls"
                        onChange={handleFileSelect}
                        disabled={isProcessing}
                        style={{ display: 'none' }}
                    />

                    {isProcessing ? (
                        <div className="component-ui-csv-uploader-processing">
                            <div className="component-ui-csv-uploader-spinner"></div>
                            <p>Processing... {uploadProgress}%</p>
                            <div className="component-ui-csv-uploader-progress-bar">
                                <div
                                    className="component-ui-csv-uploader-progress-fill"
                                    style={{ width: `${uploadProgress}%` }}
                                ></div>
                            </div>
                        </div>
                    ) : (
                        <>
                            <div className="component-ui-csv-uploader-icon">📁</div>
                            <p className="component-ui-csv-uploader-text">
                                <strong>Click to upload CSV or XLSX</strong> or drag and drop
                            </p>
                            <p className="component-ui-csv-uploader-hint">
                                Supported formats: CSV, XLS, XLSX
                            </p>
                        </>
                    )}
                </div>

                {/* Required Columns */}
                <div className="component-ui-csv-uploader-columns">
                    <h4 className="component-ui-csv-uploader-columns-title">Required Columns:</h4>
                    <div className="component-ui-csv-uploader-columns-grid">
                        {requiredColumns.map((col) => (
                            <span key={col} className="component-ui-csv-uploader-column-badge">
                                {col}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Download Template */}
                <button
                    className="component-ui-csv-uploader-template-btn"
                    onClick={() => {
                        const headers = requiredColumns.join(',');
                        const sampleData = [
                            '2025-01-15,Monday,45,10,8,75.5,45000,1000,80,60,2,2,1,5',
                            '2025-01-16,Tuesday,48,12,7,80.0,48000,1000,90,60,1,1,2,6'
                        ];
                        const csvContent = [headers, ...sampleData].join('\n');
                        const blob = new Blob([csvContent], { type: 'text/csv' });
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = 'sample_records.csv';
                        link.click();
                        URL.revokeObjectURL(url);
                    }}
                >
                    📥 Download Template
                </button>
            </div>

        </div>
    );
};

export default CSVUploader;

