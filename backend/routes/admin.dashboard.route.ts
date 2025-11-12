// @ts-ignore
import express from 'express';
import { getDashboardStats } from '../controllers/admin.dashboard.controller';
import { protect, adminOnly } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/admin/dashboard/stats:
 *   get:
 *     summary: Get admin dashboard statistics
 *     tags: [Admin Dashboard]
 *     security:
 *       - BearerAuth: []
 *     responses:
 *       200:
 *         description: Dashboard statistics retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 stats:
 *                   type: object
 *                   properties:
 *                     totalHotels:
 *                       type: number
 *                       description: Total number of hotels in the system
 *                     totalUsers:
 *                       type: number
 *                       description: Total number of users in the system
 *                     recentHotels:
 *                       type: number
 *                       description: Number of hotels added in the last 7 days
 *                     recentUsers:
 *                       type: number
 *                       description: Number of users added in the last 7 days
 *                     lastUpdated:
 *                       type: string
 *                       format: date-time
 *       401:
 *         description: Unauthorized - Admin access required
 *       500:
 *         description: Internal server error
 */
router.get('/stats', protect, adminOnly, getDashboardStats);

export default router;

