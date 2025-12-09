import React, { useState, useRef,} from 'react';
import readXlsxFile from 'read-excel-file';
import { useDispatch } from 'react-redux';
import type { AppDispatch } from '../../redux/store';
import { addRecordsToHotel } from '../../redux/services/recordsApi';

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
        try {
            const rows = await readXlsxFile(file);
            
            if (rows.length < 2) {
                throw new Error('Excel file must have at least header row and one data row');
            }

            const headers = rows[0] as string[];
            const data = rows.slice(1);

            // Convert to array of objects
            const jsonData = data.map((row) => {
                const obj: any = {};
                headers.forEach((header, index) => {
                    obj[header] = row[index];
                });
                return obj;
            });

            // Validate and convert
            return convertToRecords(jsonData);
        } catch (err: any) {
            throw new Error(`Failed to parse XLSX: ${err.message}`);
        }
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
        const isXLSX = file.name.endsWith('.xlsx');

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
        <div className="w-full">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <h3 className="text-lg font-bold text-gray-900 mb-2 flex items-center">
                    <span className="mr-2">📊</span> Upload Records via CSV/XLSX
                </h3>
                <p className="text-sm text-gray-500 mb-6">
                    Upload hotel records from a CSV or Excel file. The file must contain all required columns.
                </p>

                {/* Error/Success Messages */}
                {error && (
                    <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-700 rounded-lg text-sm">
                        {error}
                    </div>
                )}
                {success && (
                    <div className="mb-4 p-3 bg-green-50 border border-green-100 text-green-700 rounded-lg text-sm">
                        {success}
                    </div>
                )}

                {/* Upload Area */}
                <div
                    className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 group ${
                        isProcessing ? 'border-indigo-300 bg-indigo-50' : 'border-gray-300 hover:border-indigo-500 hover:bg-gray-50'
                    }`}
                    onClick={handleClick}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.xlsx"
                        onChange={handleFileSelect}
                        disabled={isProcessing}
                        style={{ display: 'none' }}
                    />

                    {isProcessing ? (
                        <div className="flex flex-col items-center justify-center">
                            <svg className="animate-spin h-8 w-8 text-indigo-600 mb-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <p className="text-sm font-medium text-indigo-600 mb-2">Processing... {uploadProgress}%</p>
                            <div className="w-full max-w-xs bg-gray-200 rounded-full h-2">
                                <div
                                    className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${uploadProgress}%` }}
                                ></div>
                            </div>
                        </div>
                    ) : (
                        <>
                            <div className="text-4xl mb-3 group-hover:scale-110 transition-transform duration-200">📁</div>
                            <p className="text-sm text-gray-900 font-medium mb-1">
                                <span className="text-indigo-600">Click to upload CSV or XLSX</span> or drag and drop
                            </p>
                            <p className="text-xs text-gray-500">
                                Supported formats: CSV, XLSX
                            </p>
                        </>
                    )}
                </div>

                {/* Required Columns */}
                <div className="mt-6">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Required Columns:</h4>
                    <div className="flex flex-wrap gap-2">
                        {requiredColumns.map((col) => (
                            <span key={col} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 border border-gray-200">
                                {col}
                            </span>
                        ))}
                    </div>
                </div>

                {/* Download Template */}
                <div className="mt-6 pt-4 border-t border-gray-100 flex justify-end">
                    <button
                        className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
                        onClick={(e) => {
                            e.stopPropagation();
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
        </div>
    );
};

export default CSVUploader;

