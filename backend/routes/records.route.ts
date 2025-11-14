// @ts-ignore
import express from 'express';
import {
    getRecentRecords,
    getRecordsByDateRange,
    getRecordsByPeriod,
    getRecordsSummary,
    getAvailableDates,
    getRevenueByPeriod,
    getRoomSoldByPeriod,
    getArrivalByPeriod,
    getDepartureByPeriod,
    getOOOByPeriod
} from '../controllers/records.controller';
import { protect } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/records/recent/{hotelId}:
 *   get:
 *     summary: Get top 5 recent records for a hotel
 *     tags: [Records]
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
 *         name: limit
 *         schema:
 *           type: number
 *           default: 5
 *         description: Number of recent records to fetch (max 100)
 *     responses:
 *       200:
 *         description: Recent records retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 count:
 *                   type: number
 *                 records:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       date:
 *                         type: string
 *                         format: date-time
 *                       roomsSold:
 *                         type: number
 *                       day:
 *                         type: string
 *                       arrivalRooms:
 *                         type: number
 *                       departureRooms:
 *                         type: number
 *                       oooRooms:
 *                         type: number
 *                       occupancyPercentage:
 *                         type: number
 *                       roomRevenue:
 *                         type: number
 *                       averageRoomRate:
 *                         type: number
 *                       pax:
 *                         type: number
 *       404:
 *         description: No records found for this hotel
 */
router.get('/recent/:hotelId', protect, getRecentRecords);

/**
 * @swagger
 * /api/v1/records/date-range:
 *   post:
 *     summary: Get records by date range
 *     tags: [Records]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - hotelId
 *               - startDate
 *               - endDate
 *             properties:
 *               hotelId:
 *                 type: string
 *                 description: Hotel ID
 *               startDate:
 *                 type: string
 *                 format: date
 *                 description: Start date (YYYY-MM-DD or ISO format)
 *               endDate:
 *                 type: string
 *                 format: date
 *                 description: End date (YYYY-MM-DD or ISO format)
 *               page:
 *                 type: number
 *                 default: 1
 *                 description: Page number for pagination
 *               limit:
 *                 type: number
 *                 default: 10
 *                 description: Number of records per page (max 100)
 *     responses:
 *       200:
 *         description: Records retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 dateRange:
 *                   type: object
 *                   properties:
 *                     startDate:
 *                       type: string
 *                     endDate:
 *                       type: string
 *                 pagination:
 *                   type: object
 *                   properties:
 *                     currentPage:
 *                       type: number
 *                     pageSize:
 *                       type: number
 *                     totalCount:
 *                       type: number
 *                     totalPages:
 *                       type: number
 *                     hasNextPage:
 *                       type: boolean
 *                     hasPreviousPage:
 *                       type: boolean
 *                 records:
 *                   type: array
 *                   items:
 *                     type: object
 *       400:
 *         description: Invalid input or date range
 */
router.post('/date-range', protect, getRecordsByDateRange);

/**
 * @swagger
 * /api/v1/records/{hotelId}:
 *   get:
 *     summary: Get records by time period
 *     description: Retrieve records for a hotel for different time periods (1 week, 1 month, 3 months, 6 months, 12 months)
 *     tags: [Records]
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
 *         description: Time period for records
 *     responses:
 *       200:
 *         description: Records retrieved successfully
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
 *                 records:
 *                   type: array
 *                   items:
 *                     type: object
 *       400:
 *         description: Invalid period or missing hotel ID
 *       404:
 *         description: No records found
 */
router.get('/:hotelId', protect, getRecordsByPeriod);

/**
 * @swagger
 * /api/v1/records/summary/{hotelId}:
 *   get:
 *     summary: Get records summary for all time periods
 *     description: Retrieve aggregated records summary (total and average values) for all time periods
 *     tags: [Records]
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
 *         description: Records summary retrieved successfully
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
 *                         avgOccupancy:
 *                           type: number
 *                         dateRange:
 *                           type: object
 *       400:
 *         description: Missing hotel ID
 */
router.get('/summary/:hotelId', protect, getRecordsSummary);

/**
 * @swagger
 * /api/v1/records/available-dates/{hotelId}:
 *   get:
 *     summary: Get available record dates for a hotel
 *     description: Retrieve all available dates for records of a hotel (useful for calendar/date picker)
 *     tags: [Records]
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
 *         description: Available dates retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 minDate:
 *                   type: string
 *                   format: date
 *                 maxDate:
 *                   type: string
 *                   format: date
 *                 count:
 *                   type: number
 *                 dates:
 *                   type: array
 *                   items:
 *                     type: string
 *                     format: date
 *       400:
 *         description: Missing hotel ID
 *       404:
 *         description: No records found for hotel
 */
router.get('/available-dates/:hotelId', protect, getAvailableDates);

/**
 * @swagger
 * /api/v1/records/revenue/{hotelId}:
 *   get:
 *     summary: Get revenue records by period
 *     description: Retrieve aggregated revenue data from actual hotel records by time period with automatic aggregation
 *     tags: [Records]
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
 *         description: Revenue records retrieved successfully
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
 *         description: No records found
 */
router.get('/revenue/:hotelId', protect, getRevenueByPeriod);

/**
 * @swagger
 * /api/v1/records/room-sold/{hotelId}:
 *   get:
 *     summary: Get room sold records by period
 *     description: Retrieve aggregated room sold data from actual hotel records by time period
 *     tags: [Records]
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
 *         description: Room sold records retrieved successfully
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
 *         description: No records found
 */
router.get('/room-sold/:hotelId', protect, getRoomSoldByPeriod);

/**
 * @swagger
 * /api/v1/records/arrival/{hotelId}:
 *   get:
 *     summary: Get arrival records by period
 *     description: Retrieve aggregated arrival room data from actual hotel records by time period
 *     tags: [Records]
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
 *         description: Arrival records retrieved successfully
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
 *         description: No records found
 */
router.get('/arrival/:hotelId', protect, getArrivalByPeriod);

/**
 * @swagger
 * /api/v1/records/departure/{hotelId}:
 *   get:
 *     summary: Get departure records by period
 *     description: Retrieve aggregated departure room data from actual hotel records by time period
 *     tags: [Records]
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
 *         description: Departure records retrieved successfully
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
 *         description: No records found
 */
router.get('/departure/:hotelId', protect, getDepartureByPeriod);

/**
 * @swagger
 * /api/v1/records/ooo/{hotelId}:
 *   get:
 *     summary: Get OOO (Out of Order) records by period
 *     description: Retrieve aggregated OOO room data from actual hotel records by time period
 *     tags: [Records]
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
 *         description: OOO records retrieved successfully
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
 *         description: No records found
 */
router.get('/ooo/:hotelId', protect, getOOOByPeriod);

export default router;

