import { Request, Response } from 'express';
import Record from '../models/Record';
import {getPeriodConfig, aggregateMetric} from '../helpers/record.forcast.helper';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

/**
 * Get all available record dates for a hotel
 */
export const getAvailableDates = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const records = await Record.find({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const dates = records.map(record => ({
            date: new Date(record.date).toISOString().split('T')[0], // Format as YYYY-MM-DD
            timestamp: record.date
        }));

        const minDate = records[0].date;
        const maxDate = records[records.length - 1].date;

        res.status(200).json({
            message: 'Available dates retrieved successfully',
            minDate: minDate.toISOString().split('T')[0],
            maxDate: maxDate.toISOString().split('T')[0],
            count: dates.length,
            dates: dates.map(d => d.date)
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const getRecentRecords = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { limit = 5 } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const parsedLimit = Math.min(parseInt(limit as string) || 5, 100);

        const records = await Record.find({ hotelId })
            .sort({ date: -1 })
            .limit(parsedLimit)
            .select(
                'hotelId date roomsSold day arrivalRooms departureRooms oooRooms occupancyPercentage roomRevenue averageRoomRate pax complimentRooms houseUse individualConfirm totalRoomInventory'
            );

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const formattedRecords = records.map(record => ({
            id: record._id,
            date: record.date,
            roomsSold: record.roomsSold,
            day: record.day,
            arrivalRooms: record.arrivalRooms,
            departureRooms: record.departureRooms,
            oooRooms: record.oooRooms,
            occupancyPercentage: record.occupancyPercentage,
            roomRevenue: record.roomRevenue,
            averageRoomRate: record.averageRoomRate,
            pax: record.pax,
            complimentRooms: record.complimentRooms,
            houseUse: record.houseUse,
            individualConfirm: record.individualConfirm,
            totalRoomInventory: record.totalRoomInventory
        }));

        res.status(200).json({
            message: 'Recent records retrieved successfully',
            count: formattedRecords.length,
            records: formattedRecords
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get records by time period
 */
export const getRecordsByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const validPeriods = ['1w', '1m', '3m', '6m', '12m'];
        if (!validPeriods.includes(period as string)) {
            throw new ApiError('Invalid period. Valid periods are: 1w, 1m, 3m, 6m, 12m', 400);
        }

        // Find the last (most recent) record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        // Use the last record date as the reference point (endDate)
        const endDate = new Date(lastRecord.date);
        let startDate = new Date(endDate);

        // Calculate start date based on the period
        switch (period) {
            case '1w':
                startDate.setDate(endDate.getDate() - 7);
                break;
            case '1m':
                startDate.setMonth(endDate.getMonth() - 1);
                break;
            case '3m':
                startDate.setMonth(endDate.getMonth() - 3);
                break;
            case '6m':
                startDate.setMonth(endDate.getMonth() - 6);
                break;
            case '12m':
                startDate.setFullYear(endDate.getFullYear() - 1);
                break;
        }

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate, $lte: endDate }
        })
            .sort({ date: -1 })
            .select(
                'hotelId date roomsSold day arrivalRooms departureRooms oooRooms occupancyPercentage roomRevenue averageRoomRate pax complimentRooms houseUse individualConfirm totalRoomInventory'
            );

        if (!records || records.length === 0) {
            throw new ApiError(`No records found for this hotel for the ${period} period`, 404);
        }

        const formattedRecords = records.map(record => ({
            id: record._id,
            date: record.date,
            roomsSold: record.roomsSold,
            day: record.day,
            arrivalRooms: record.arrivalRooms,
            departureRooms: record.departureRooms,
            oooRooms: record.oooRooms,
            occupancyPercentage: record.occupancyPercentage,
            roomRevenue: record.roomRevenue,
            averageRoomRate: record.averageRoomRate,
            pax: record.pax,
            complimentRooms: record.complimentRooms,
            houseUse: record.houseUse,
            individualConfirm: record.individualConfirm,
            totalRoomInventory: record.totalRoomInventory
        }));

        res.status(200).json({
            message: `Records retrieved successfully for ${period} period`,
            period,
            referenceDate: endDate.toISOString(),
            dateRange: {
                startDate: startDate.toISOString(),
                endDate: endDate.toISOString()
            },
            count: formattedRecords.length,
            records: formattedRecords
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get records summary for different time periods
 */
export const getRecordsSummary = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Find the last (most recent) record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        // Use the last record date as the reference point
        const endDate = new Date(lastRecord.date);
        const periods = ['1w', '1m', '3m', '6m', '12m'];
        const summaryData: any = {};

        for (const period of periods) {
            let startDate = new Date(endDate);

            switch (period) {
                case '1w':
                    startDate.setDate(endDate.getDate() - 7);
                    break;
                case '1m':
                    startDate.setMonth(endDate.getMonth() - 1);
                    break;
                case '3m':
                    startDate.setMonth(endDate.getMonth() - 3);
                    break;
                case '6m':
                    startDate.setMonth(endDate.getMonth() - 6);
                    break;
                case '12m':
                    startDate.setFullYear(endDate.getFullYear() - 1);
                    break;
            }

            const records = await Record.find({
                hotelId,
                date: { $gte: startDate, $lte: endDate }
            });

            const totalRevenue = records.reduce((sum, r) => sum + r.roomRevenue, 0);
            const totalRoomsSold = records.reduce((sum, r) => sum + r.roomsSold, 0);
            const totalComplimentRooms = records.reduce((sum, r) => sum + r.complimentRooms, 0);
            const totalHouseUse = records.reduce((sum, r) => sum + r.houseUse, 0);
            const totalIndividualConfirm = records.reduce((sum, r) => sum + r.individualConfirm, 0);
            const totalRoomInventory = records.reduce((sum, r) => sum + r.totalRoomInventory, 0);

            const avgRevenue = records.length > 0 ? Math.round(totalRevenue / records.length) : 0;
            const avgRoomsSold = records.length > 0 ? Math.round(totalRoomsSold / records.length) : 0;
            const avgOccupancy = records.length > 0
                ? Math.round((records.reduce((sum, r) => sum + r.occupancyPercentage, 0) / records.length) * 100) / 100
                : 0;
            const avgComplimentRooms = records.length > 0 ? Math.round((totalComplimentRooms / records.length) * 100) / 100 : 0;
            const avgHouseUse = records.length > 0 ? Math.round((totalHouseUse / records.length) * 100) / 100 : 0;
            const avgIndividualConfirm = records.length > 0 ? Math.round((totalIndividualConfirm / records.length) * 100) / 100 : 0;
            const avgRoomInventory = records.length > 0 ? Math.round((totalRoomInventory / records.length) * 100) / 100 : 0;

            summaryData[period] = {
                count: records.length,
                totalRevenue,
                totalRoomsSold,
                totalComplimentRooms,
                totalHouseUse,
                totalIndividualConfirm,
                totalRoomInventory,
                avgRevenue,
                avgRoomsSold,
                avgOccupancy,
                avgComplimentRooms,
                avgHouseUse,
                avgIndividualConfirm,
                avgRoomInventory,
                dateRange: {
                    startDate,
                    endDate
                }
            };
        }

        res.status(200).json({
            message: 'Records summary retrieved successfully',
            hotelId,
            referenceDate: endDate.toISOString(),
            summary: summaryData
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const getRecordsByDateRange = async (req: Request, res: Response) => {
    try {
        const { hotelId, startDate, endDate, page = 1, limit = 10 } = req.body;

        if (!hotelId || !startDate || !endDate) {
            throw new ApiError('Hotel ID, start date, and end date are required', 400);
        }

        // Validate date format
        const start = new Date(startDate);
        const end = new Date(endDate);

        if (isNaN(start.getTime()) || isNaN(end.getTime())) {
            throw new ApiError('Invalid date format. Use ISO 8601 format (YYYY-MM-DD)', 400);
        }

        if (start > end) {
            throw new ApiError('Start date must be before end date', 400);
        }

        const pageNum = Math.max(1, parseInt(page) || 1);
        const limitNum = Math.min(parseInt(limit) || 10, 100);
        const skip = (pageNum - 1) * limitNum;

        // Get total count for pagination
        const totalCount = await Record.countDocuments({
            hotelId,
            date: { $gte: start, $lte: end }
        });

        const records = await Record.find({
            hotelId,
            date: { $gte: start, $lte: end }
        })
            .sort({ date: -1 })
            .skip(skip)
            .limit(limitNum)
            .select(
                'hotelId date roomsSold day arrivalRooms departureRooms oooRooms occupancyPercentage roomRevenue averageRoomRate pax complimentRooms houseUse individualConfirm totalRoomInventory'
            );

        const formattedRecords = records.map(record => ({
            id: record._id,
            date: record.date,
            roomsSold: record.roomsSold,
            day: record.day,
            arrivalRooms: record.arrivalRooms,
            departureRooms: record.departureRooms,
            oooRooms: record.oooRooms,
            occupancyPercentage: record.occupancyPercentage,
            roomRevenue: record.roomRevenue,
            averageRoomRate: record.averageRoomRate,
            pax: record.pax,
            complimentRooms: record.complimentRooms,
            houseUse: record.houseUse,
            individualConfirm: record.individualConfirm,
            totalRoomInventory: record.totalRoomInventory
        }));

        res.status(200).json({
            message: 'Records retrieved successfully',
            dateRange: {
                startDate,
                endDate
            },
            pagination: {
                currentPage: pageNum,
                pageSize: limitNum,
                totalCount,
                totalPages: Math.ceil(totalCount / limitNum),
                hasNextPage: skip + limitNum < totalCount,
                hasPreviousPage: pageNum > 1
            },
            records: formattedRecords
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};





/**
 * Get Revenue data by period
 */
export const getRevenueByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the last record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const { startDate, aggregationDays } = getPeriodConfig(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastRecord.date));

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate }
        })
            .sort({ date: 1 })
            .select('date roomRevenue');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const aggregated = aggregateMetric(records, 'roomRevenue', aggregationDays);

        res.status(200).json({
            message: 'Revenue data retrieved successfully',
            period,
            referenceDate: lastRecord.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get Room Sold data by period
 */
export const getRoomSoldByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the last record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const { startDate, aggregationDays } = getPeriodConfig(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastRecord.date));

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate }
        })
            .sort({ date: 1 })
            .select('date roomsSold');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const aggregated = aggregateMetric(records, 'roomsSold', aggregationDays);

        res.status(200).json({
            message: 'Room sold data retrieved successfully',
            period,
            referenceDate: lastRecord.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get Arrival Rooms data by period
 */
export const getArrivalByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the last record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const { startDate, aggregationDays } = getPeriodConfig(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastRecord.date));

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate }
        })
            .sort({ date: 1 })
            .select('date arrivalRooms');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const aggregated = aggregateMetric(records, 'arrivalRooms', aggregationDays);

        res.status(200).json({
            message: 'Arrival data retrieved successfully',
            period,
            referenceDate: lastRecord.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get Departure Rooms data by period
 */
export const getDepartureByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the last record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const { startDate, aggregationDays } = getPeriodConfig(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastRecord.date));

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate }
        })
            .sort({ date: 1 })
            .select('date departureRooms');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const aggregated = aggregateMetric(records, 'departureRooms', aggregationDays);

        res.status(200).json({
            message: 'Departure data retrieved successfully',
            period,
            referenceDate: lastRecord.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get OOO Rooms data by period
 */
export const getOOOByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the last record date for this hotel
        const lastRecord = await Record.findOne({ hotelId })
            .sort({ date: -1 })
            .select('date');

        if (!lastRecord) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const { startDate, aggregationDays } = getPeriodConfig(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastRecord.date));

        const records = await Record.find({
            hotelId,
            date: { $gte: startDate }
        })
            .sort({ date: 1 })
            .select('date oooRooms');

        if (!records || records.length === 0) {
            throw new ApiError('No records found for this hotel', 404);
        }

        const aggregated = aggregateMetric(records, 'oooRooms', aggregationDays);

        res.status(200).json({
            message: 'OOO data retrieved successfully',
            period,
            referenceDate: lastRecord.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};
