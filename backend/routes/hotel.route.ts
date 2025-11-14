// @ts-ignore
import express from 'express';
import { addHotel, getRecentlyAddedHotels, searchHotels, getAllHotels, getHotelGeneralInfo } from '../controllers/hotel.controller';
import { protect, adminOnly } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/hotels/add_hotel:
 *   post:
 *     summary: Add a new hotel
 *     tags: [Hotels]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - name
 *               - email
 *               - contactNumber
 *               - plotNo
 *               - streetName
 *               - city
 *               - state
 *               - pincode
 *             properties:
 *               name:
 *                 type: string
 *                 description: Name of the hotel
 *               email:
 *                 type: string
 *                 format: email
 *                 description: Contact email for the hotel
 *               contactNumber:
 *                 type: string
 *                 description: Contact phone number
 *               plotNo:
 *                 type: string
 *                 description: Plot number of the hotel
 *               streetName:
 *                 type: string
 *                 description: Street name where hotel is located
 *               city:
 *                 type: string
 *                 description: City where hotel is located
 *               state:
 *                 type: string
 *                 description: State where hotel is located
 *               pincode:
 *                 type: string
 *                 description: Postal code of the hotel location
 *     responses:
 *       201:
 *         description: Hotel added successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 hotel:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     name:
 *                       type: string
 *                     email:
 *                       type: string
 *                     city:
 *                       type: string
 *                     state:
 *                       type: string
 *       400:
 *         description: Invalid input or hotel already exists
 *       403:
 *         description: Unauthorized - Admin access required
 */
router.post('/add_hotel', protect, adminOnly, addHotel);

/**
 * @swagger
 * /api/v1/hotels/recently-added:
 *   post:
 *     summary: Get recently added hotels
 *     tags: [Hotels]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               limit:
 *                 type: number
 *                 default: 5
 *                 description: Number of recently added hotels to fetch (max 100)
 *     responses:
 *       200:
 *         description: Recently added hotels retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 count:
 *                   type: number
 *                 hotels:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       hotelName:
 *                         type: string
 *                       ownerName:
 *                         type: string
 *                       city:
 *                         type: string
 *                       contactNumber:
 *                         type: string
 *                       imageUrl:
 *                         type: string
 *                       addedAt:
 *                         type: string
 *                         format: date-time
 *       400:
 *         description: Invalid input
 */
router.post('/recently-added', protect, getRecentlyAddedHotels);

/**
 * @swagger
 * /api/v1/hotels/search:
 *   post:
 *     summary: Search hotels by name or owner name
 *     tags: [Hotels]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - searchTerm
 *             properties:
 *               searchTerm:
 *                 type: string
 *                 description: Search term for hotel name or owner name
 *               page:
 *                 type: number
 *                 default: 1
 *                 description: Page number for pagination
 *               limit:
 *                 type: number
 *                 default: 10
 *                 description: Number of results per page (max 100)
 *     responses:
 *       200:
 *         description: Hotels found
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 searchTerm:
 *                   type: string
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
 *                 hotels:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       hotelName:
 *                         type: string
 *                       ownerName:
 *                         type: string
 *                       city:
 *                         type: string
 *                       contactNumber:
 *                         type: string
 *                       imageUrl:
 *                         type: string
 *       400:
 *         description: Missing search term
 */
router.post('/search', protect, searchHotels);

/**
 * @swagger
 * /api/v1/hotels:
 *   get:
 *     summary: Get all hotels with pagination (for infinite scroll)
 *     tags: [Hotels]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: query
 *         name: page
 *         schema:
 *           type: number
 *           default: 1
 *         description: Page number for pagination
 *       - in: query
 *         name: limit
 *         schema:
 *           type: number
 *           default: 10
 *         description: Number of results per page (max 100)
 *     responses:
 *       200:
 *         description: Hotels retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
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
 *                 hotels:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       name:
 *                         type: string
 *                       email:
 *                         type: string
 *                       contactNumber:
 *                         type: string
 *                       city:
 *                         type: string
 *                       state:
 *                         type: string
 *                       imageUrl:
 *                         type: string
 *                       adminName:
 *                         type: string
 */
router.get('/', protect, getAllHotels);

/**
 * @swagger
 * /api/v1/hotels/general-info/{hotelId}:
 *   get:
 *     summary: Get hotel general information
 *     tags: [Hotels]
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
 *         description: Hotel general information retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 hotel:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     name:
 *                       type: string
 *                     email:
 *                       type: string
 *                     contactNumber:
 *                       type: string
 *                     plotNo:
 *                       type: string
 *                     streetName:
 *                       type: string
 *                     city:
 *                       type: string
 *                     state:
 *                       type: string
 *                     pincode:
 *                       type: string
 *       404:
 *         description: Hotel not found
 *       400:
 *         description: Invalid input
 */
router.get('/general-info/:hotelId', protect, getHotelGeneralInfo);

export default router;
