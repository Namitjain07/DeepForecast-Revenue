// @ts-ignore
import express from 'express';
import { addLastTrain, getLastTrainByHotel } from '../controllers/train.controller';
import { protect } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/train/add:
 *   post:
 *     summary: Add a new last train record
 *     tags: [Train]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - userId
 *               - hotelId
 *             properties:
 *               userId:
 *                 type: string
 *                 description: User ID
 *               hotelId:
 *                 type: string
 *                 description: Hotel ID
 *     responses:
 *       201:
 *         description: Last train record created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 lastTrain:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     hotelId:
 *                       type: string
 *                     userId:
 *                       type: string
 *                     startDateTime:
 *                       type: string
 *                       format: date-time
 *                     endDateTime:
 *                       type: string
 *                       format: date-time
 *                     status:
 *                       type: string
 *                       enum: ['none', 'queued', 'running', 'success', 'failure']
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 *                     updatedAt:
 *                       type: string
 *                       format: date-time
 *       400:
 *         description: Invalid input (missing userId or hotelId)
 *       500:
 *         description: Server error
 */
router.post('/start', protect, addLastTrain);

/**
 * @swagger
 * /api/v1/train/{hotelId}:
 *   get:
 *     summary: Get last train record by hotel ID
 *     tags: [Train]
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
 *         description: Last train record retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 lastTrain:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     hotelId:
 *                       type: string
 *                     userId:
 *                       type: string
 *                     startDateTime:
 *                       type: string
 *                       format: date-time
 *                     endDateTime:
 *                       type: string
 *                       format: date-time
 *                     status:
 *                       type: string
 *                       enum: ['none', 'queued', 'running', 'success', 'failure']
 *                     createdAt:
 *                       type: string
 *                       format: date-time
 *                     updatedAt:
 *                       type: string
 *                       format: date-time
 *       404:
 *         description: No train record found for this hotel
 *       400:
 *         description: Hotel ID is required
 *       500:
 *         description: Server error
 */
router.get('/:hotelId', protect, getLastTrainByHotel);

export default router;

