import { Request, Response } from 'express';
import Forecast from '../models/Forecast';
import {getPeriodConfigForecast, aggregateMetric} from '../helpers/record.forcast.helper';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

/**
 * Get all available forecast dates for a hotel
 */
export const getAvailableDates = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const forecasts = await Forecast.find({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const dates = forecasts.map(forecast =>
            new Date(forecast.date).toISOString().split('T')[0]
        );

        const minDate = forecasts[0]!.date;
        const maxDate = forecasts[forecasts.length - 1]!.date;

        res.status(200).json({
            message: 'Available forecast dates retrieved successfully',
            minDate: new Date(minDate).toISOString().split('T')[0],
            maxDate: new Date(maxDate).toISOString().split('T')[0],
            count: dates.length,
            dates
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get forecast data for different time periods
 */
export const getForecastByPeriod = async (req: Request, res: Response) => {
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

        // Calculate date range based on period
        const today = new Date();
        let startDate = new Date();
        let endDate = new Date();

        switch (period) {
            case '1w':
                startDate.setDate(today.getDate() - 7);
                endDate.setDate(today.getDate() + 7);
                break;
            case '1m':
                startDate.setMonth(today.getMonth() - 1);
                endDate.setMonth(today.getMonth() + 1);
                break;
            case '3m':
                startDate.setMonth(today.getMonth() - 3);
                endDate.setMonth(today.getMonth() + 3);
                break;
            case '6m':
                startDate.setMonth(today.getMonth() - 6);
                endDate.setMonth(today.getMonth() + 6);
                break;
            case '12m':
                startDate.setFullYear(today.getFullYear() - 1);
                endDate.setFullYear(today.getFullYear() + 1);
                break;
        }

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: startDate, $lte: endDate }
        })
            .sort({ date: 1 })
            .select('hotelId date revenue roomSold arrivalRoom departureRoom oooRoom');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError(`No forecast data found for this hotel for the ${period} period`, 404);
        }

        const formattedForecasts = forecasts.map(forecast => ({
            id: forecast._id,
            date: forecast.date,
            revenue: forecast.revenue,
            roomSold: forecast.roomSold,
            arrivalRoom: forecast.arrivalRoom,
            departureRoom: forecast.departureRoom,
            oooRoom: forecast.oooRoom
        }));

        res.status(200).json({
            message: `Forecast data retrieved successfully for ${period} period`,
            period,
            dateRange: {
                startDate: startDate.toISOString(),
                endDate: endDate.toISOString()
            },
            count: formattedForecasts.length,
            forecasts: formattedForecasts
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get forecast summary for different time periods
 */
export const getForecastSummary = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const periods = ['1w', '1m', '3m', '6m', '12m'];
        const summaryData: any = {};

        for (const period of periods) {
            const today = new Date();
            let startDate = new Date();
            let endDate = new Date();

            switch (period) {
                case '1w':
                    startDate.setDate(today.getDate() - 7);
                    endDate.setDate(today.getDate() + 7);
                    break;
                case '1m':
                    startDate.setMonth(today.getMonth() - 1);
                    endDate.setMonth(today.getMonth() + 1);
                    break;
                case '3m':
                    startDate.setMonth(today.getMonth() - 3);
                    endDate.setMonth(today.getMonth() + 3);
                    break;
                case '6m':
                    startDate.setMonth(today.getMonth() - 6);
                    endDate.setMonth(today.getMonth() + 6);
                    break;
                case '12m':
                    startDate.setFullYear(today.getFullYear() - 1);
                    endDate.setFullYear(today.getFullYear() + 1);
                    break;
            }

            const forecasts = await Forecast.find({
                hotelId,
                date: { $gte: startDate, $lte: endDate }
            });

            const totalRevenue = forecasts.reduce((sum, f) => sum + f.revenue, 0);
            const totalRoomsSold = forecasts.reduce((sum, f) => sum + f.roomSold, 0);
            const avgRevenue = forecasts.length > 0 ? Math.round(totalRevenue / forecasts.length) : 0;
            const avgRoomsSold = forecasts.length > 0 ? Math.round(totalRoomsSold / forecasts.length) : 0;

            summaryData[period] = {
                count: forecasts.length,
                totalRevenue,
                totalRoomsSold,
                avgRevenue,
                avgRoomsSold,
                dateRange: {
                    startDate,
                    endDate
                }
            };
        }

        res.status(200).json({
            message: 'Forecast summary retrieved successfully',
            hotelId,
            summary: summaryData
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get forecast data by date range
 */
export const getDateRangeForecasts = async (req: Request, res: Response) => {
    try {
        const { hotelId, startDate, endDate, page = 1, limit = 10 } = req.body;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        if (!startDate || !endDate) {
            throw new ApiError('Start date and end date are required', 400);
        }

        const start = new Date(startDate);
        const end = new Date(endDate);

        if (start > end) {
            throw new ApiError('Start date cannot be after end date', 400);
        }

        const skip = (parseInt(page as any) - 1) * parseInt(limit as any);
        const limitNum = Math.min(parseInt(limit as any), 100);

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: start, $lte: end }
        })
            .sort({ date: -1 })
            .skip(skip)
            .limit(limitNum)
            .select('hotelId date revenue roomSold arrivalRoom departureRoom oooRoom');

        const totalCount = await Forecast.countDocuments({
            hotelId,
            date: { $gte: start, $lte: end }
        });

        const formattedForecasts = forecasts.map(forecast => ({
            id: forecast._id,
            date: forecast.date,
            revenue: forecast.revenue,
            roomSold: forecast.roomSold,
            arrivalRoom: forecast.arrivalRoom,
            departureRoom: forecast.departureRoom,
            oooRoom: forecast.oooRoom
        }));

        res.status(200).json({
            message: 'Forecast data retrieved successfully',
            dateRange: {
                startDate: start.toISOString(),
                endDate: end.toISOString()
            },
            pagination: {
                currentPage: parseInt(page as any),
                pageSize: limitNum,
                totalCount,
                totalPages: Math.ceil(totalCount / limitNum),
                hasNextPage: skip + limitNum < totalCount,
                hasPreviousPage: skip > 0
            },
            count: formattedForecasts.length,
            forecasts: formattedForecasts
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};



/**
 * Get Revenue forecast by period (forward from last forecast date)
 */
export const getRevenueForcastByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the first forecast date for this hotel
        const lastForecast = await Forecast.findOne({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!lastForecast) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const { endDate, aggregationDays } = getPeriodConfigForecast(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastForecast.date));

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: new Date(lastForecast.date), $lte: endDate }
        })
            .sort({ date: 1 })
            .select('date revenue');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const aggregated = aggregateMetric(forecasts, 'revenue', aggregationDays);

        res.status(200).json({
            message: 'Revenue forecast retrieved successfully',
            period,
            referenceDate: lastForecast.date.toISOString(),
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
 * Get Room Sold forecast by period (forward from last forecast date)
 */
export const getRoomSoldForcastByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the first forecast date for this hotel
        const lastForecast = await Forecast.findOne({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!lastForecast) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const { endDate, aggregationDays } = getPeriodConfigForecast(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastForecast.date));

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: new Date(lastForecast.date), $lte: endDate }
        })
            .sort({ date: 1 })
            .select('date roomSold');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const aggregated = aggregateMetric(forecasts, 'roomSold', aggregationDays);

        res.status(200).json({
            message: 'Room sold forecast retrieved successfully',
            period,
            referenceDate: lastForecast.date.toISOString(),
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
 * Get Arrival forecast by period (forward from last forecast date)
 */
export const getArrivalForcastByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the first forecast date for this hotel
        const lastForecast = await Forecast.findOne({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!lastForecast) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const { endDate, aggregationDays } = getPeriodConfigForecast(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastForecast.date));

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: new Date(lastForecast.date), $lte: endDate }
        })
            .sort({ date: 1 })
            .select('date arrivalRoom');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const aggregated = aggregateMetric(forecasts, 'arrivalRoom', aggregationDays);

        res.status(200).json({
            message: 'Arrival forecast retrieved successfully',
            period,
            referenceDate: lastForecast.date.toISOString(),
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
 * Get Departure forecast by period (forward from last forecast date)
 */
export const getDepartureForcastByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the first forecast date for this hotel
        const lastForecast = await Forecast.findOne({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!lastForecast) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const { endDate, aggregationDays } = getPeriodConfigForecast(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastForecast.date));

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: new Date(lastForecast.date), $lte: endDate }
        })
            .sort({ date: 1 })
            .select('date departureRoom');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const aggregated = aggregateMetric(forecasts, 'departureRoom', aggregationDays);

        res.status(200).json({
            message: 'Departure forecast retrieved successfully',
            period,
            referenceDate: lastForecast.date.toISOString(),
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
 * Get OOO forecast by period (forward from last forecast date)
 */
export const getOOOForcastByPeriod = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { period = '1m' } = req.query;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Get the first forecast date for this hotel
        const lastForecast = await Forecast.findOne({ hotelId })
            .sort({ date: 1 })
            .select('date');

        if (!lastForecast) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const { endDate, aggregationDays } = getPeriodConfigForecast(period as '1w' | '1m' | '3m' | '6m' | '12m', new Date(lastForecast.date));

        const forecasts = await Forecast.find({
            hotelId,
            date: { $gte: new Date(lastForecast.date), $lte: endDate }
        })
            .sort({ date: 1 })
            .select('date oooRoom');

        if (!forecasts || forecasts.length === 0) {
            throw new ApiError('No forecast data found for this hotel', 404);
        }

        const aggregated = aggregateMetric(forecasts, 'oooRoom', aggregationDays);

        res.status(200).json({
            message: 'OOO forecast retrieved successfully',
            period,
            referenceDate: lastForecast.date.toISOString(),
            data: aggregated
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const getSingleDayForecast = async (req: Request, res: Response) => {
    try {
        const { hotelId, date } = req.params;

        if (!hotelId || !date) {
            throw new ApiError('Hotel ID and date are required', 400);
        }

        // Validate date format
        const forecastDate = new Date(date);
        if (isNaN(forecastDate.getTime())) {
            throw new ApiError('Invalid date format. Use ISO 8601 format (YYYY-MM-DD)', 400);
        }

        // Set the date to start of day
        const startOfDay = new Date(forecastDate);
        startOfDay.setHours(0, 0, 0, 0);

        // Set the date to end of day
        // const endOfDay = new Date(forecastDate);
        // endOfDay.setHours(23, 59, 59, 999);

        const forecast = await Forecast.findOne({
            hotelId,
            date: { $gte: startOfDay}
        }).select(
            'hotelId date revenue roomSold arrivalRoom departureRoom oooRoom'
        );

        if (!forecast) {
            throw new ApiError('No forecast data found for this date', 404);
        }

        res.status(200).json({
            message: 'Forecast data retrieved successfully',
            forecast: {
                id: forecast._id,
                hotelId: forecast.hotelId,
                date: forecast.date,
                revenue: forecast.revenue,
                roomSold: forecast.roomSold,
                arrivalRoom: forecast.arrivalRoom,
                departureRoom: forecast.departureRoom,
                oooRoom: forecast.oooRoom
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};
