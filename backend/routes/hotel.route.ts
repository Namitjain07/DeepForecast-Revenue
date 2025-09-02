// @ts-ignore
import express from 'express';
import { addHotel } from '../controllers/hotel.controller';
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

export default router;
