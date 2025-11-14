// @ts-ignore
import express from 'express';
import {
    getForecastByPeriod,
    getForecastSummary,
    getAvailableDates,
    getDateRangeForecasts,
    getRevenueForcastByPeriod,
    getRoomSoldForcastByPeriod,
    getArrivalForcastByPeriod,
    getDepartureForcastByPeriod,
    getOOOForcastByPeriod,
    getSingleDayForecast
} from '../controllers/forcast.controller';
import { protect } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/forecast/{hotelId}:
 *   get:
 *     summary: Get forecast data by time period
 *     description: Retrieve forecast data for a hotel for different time periods (1 week, 1 month, 3 months, 6 months, 12 months)
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period for forecast data
 *     responses:
 *       200:
 *         description: Forecast data retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 dateRange:
 *                   type: object
 *                   properties:
 *                     startDate:
 *                       type: string
 *                       format: date-time
 *                     endDate:
 *                       type: string
 *                       format: date-time
 *                 count:
 *                   type: number
 *                 forecasts:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       revenue:
 *                         type: number
 *                       roomSold:
 *                         type: number
 *                       arrivalRoom:
 *                         type: number
 *                       departureRoom:
 *                         type: number
 *                       oooRoom:
 *                         type: number
 *       400:
 *         description: Invalid period or missing hotel ID
 *       404:
 *         description: No forecast data found
 */
router.get('/:hotelId', protect, getForecastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/summary/{hotelId}:
 *   get:
 *     summary: Get forecast summary for all time periods
 *     description: Retrieve aggregated forecast summary (total and average values) for all time periods
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *     responses:
 *       200:
 *         description: Forecast summary retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 hotelId:
 *                   type: string
 *                 summary:
 *                   type: object
 *                   properties:
 *                     "1w":
 *                       type: object
 *                       properties:
 *                         count:
 *                           type: number
 *                         totalRevenue:
 *                           type: number
 *                         totalRoomsSold:
 *                           type: number
 *                         avgRevenue:
 *                           type: number
 *                         avgRoomsSold:
 *                           type: number
 *                         dateRange:
 *                           type: object
 *       400:
 *         description: Missing hotel ID
 */
router.get('/summary/:hotelId', protect, getForecastSummary);

/**
 * @swagger
 * /api/v1/forecast/available-dates/{hotelId}:
 *   get:
 *     summary: Get available forecast dates for a hotel
 *     description: Retrieve all available dates for forecasts of a hotel
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *     responses:
 *       200:
 *         description: Available forecast dates retrieved successfully
 */
router.get('/available-dates/:hotelId', protect, getAvailableDates);

/**
 * @swagger
 * /api/v1/forecast/date-range:
 *   post:
 *     summary: Get forecast data by date range
 *     description: Retrieve forecast data for a date range with pagination
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               hotelId:
 *                 type: string
 *               startDate:
 *                 type: string
 *                 format: date
 *               endDate:
 *                 type: string
 *                 format: date
 *               page:
 *                 type: number
 *                 default: 1
 *               limit:
 *                 type: number
 *                 default: 10
 *     responses:
 *       200:
 *         description: Forecast data retrieved successfully
 */
router.post('/date-range', protect, getDateRangeForecasts);

/**
 * @swagger
 * /api/v1/forecast/revenue/{hotelId}:
 *   get:
 *     summary: Get revenue forecast by period
 *     description: Retrieve aggregated revenue forecast data for a hotel by time period with automatic aggregation
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period (1w=daily, 1m=daily, 3m=3-day, 6m=weekly, 12m=monthly)
 *     responses:
 *       200:
 *         description: Revenue forecast retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       value:
 *                         type: number
 *                         description: Aggregated revenue value
 *       400:
 *         description: Invalid period or missing hotel ID
 *       404:
 *         description: No forecast data found
 */
router.get('/revenue/:hotelId', protect, getRevenueForcastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/room-sold/{hotelId}:
 *   get:
 *     summary: Get room sold forecast by period
 *     description: Retrieve aggregated room sold forecast data for a hotel by time period
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period (1w=daily, 1m=daily, 3m=3-day, 6m=weekly, 12m=monthly)
 *     responses:
 *       200:
 *         description: Room sold forecast retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       value:
 *                         type: number
 *       404:
 *         description: No forecast data found
 */
router.get('/room-sold/:hotelId', protect, getRoomSoldForcastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/arrival/{hotelId}:
 *   get:
 *     summary: Get arrival forecast by period
 *     description: Retrieve aggregated arrival room forecast data for a hotel by time period
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period (1w=daily, 1m=daily, 3m=3-day, 6m=weekly, 12m=monthly)
 *     responses:
 *       200:
 *         description: Arrival forecast retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       value:
 *                         type: number
 *       404:
 *         description: No forecast data found
 */
router.get('/arrival/:hotelId', protect, getArrivalForcastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/departure/{hotelId}:
 *   get:
 *     summary: Get departure forecast by period
 *     description: Retrieve aggregated departure room forecast data for a hotel by time period
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period (1w=daily, 1m=daily, 3m=3-day, 6m=weekly, 12m=monthly)
 *     responses:
 *       200:
 *         description: Departure forecast retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       value:
 *                         type: number
 *       404:
 *         description: No forecast data found
 */
router.get('/departure/:hotelId', protect, getDepartureForcastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/ooo/{hotelId}:
 *   get:
 *     summary: Get OOO (Out of Order) forecast by period
 *     description: Retrieve aggregated OOO room forecast data for a hotel by time period
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: ['1w', '1m', '3m', '6m', '12m']
 *           default: '1m'
 *         required: false
 *         description: Time period (1w=daily, 1m=daily, 3m=3-day, 6m=weekly, 12m=monthly)
 *     responses:
 *       200:
 *         description: OOO forecast retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 period:
 *                   type: string
 *                 data:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       value:
 *                         type: number
 *       404:
 *         description: No forecast data found
 */
router.get('/ooo/:hotelId', protect, getOOOForcastByPeriod);

/**
 * @swagger
 * /api/v1/forecast/day/{hotelId}/{date}:
 *   get:
 *     summary: Get forecast data for a specific day
 *     description: Retrieve detailed forecast data for a hotel on a specific date
 *     tags: [Forecast]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: hotelId
 *         schema:
 *           type: string
 *         required: true
 *         description: Hotel ID
 *       - in: path
 *         name: date
 *         schema:
 *           type: string
 *           format: date
 *         required: true
 *         description: Date in YYYY-MM-DD format
 *     responses:
 *       200:
 *         description: Forecast data retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 forecast:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     hotelId:
 *                       type: string
 *                     date:
 *                       type: string
 *                       format: date-time
 *                     revenue:
 *                       type: number
 *                     roomSold:
 *                       type: number
 *                     arrivalRoom:
 *                       type: number
 *                     departureRoom:
 *                       type: number
 *                     oooRoom:
 *                       type: number
 *
 *       404:
 *         description: No forecast data found for this date
 *       400:
 *         description: Invalid input
 */
router.get('/day/:hotelId/:date', protect, getSingleDayForecast);

export default router;

