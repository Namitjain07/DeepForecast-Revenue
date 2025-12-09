// @ts-ignore
import express from 'express';
import {
    getUsersByHotel,
    getUserById,
    updateUser,
    deleteUser
} from '../controllers/user.controller';
import { protect } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/users/hotel/{hotelId}:
 *   get:
 *     summary: Get all users for a specific hotel
 *     tags: [Users]
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
 *         description: Users retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 count:
 *                   type: number
 *                 users:
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
 *                       role:
 *                         type: string
 *                         enum: ['owner', 'manager']
 *       404:
 *         description: No users found for this hotel
 */
router.get('/hotel/:hotelId', protect, getUsersByHotel);

/**
 * @swagger
 * /api/v1/users/{userId}:
 *   get:
 *     summary: Get user by ID
 *     tags: [Users]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         schema:
 *           type: string
 *         required: true
 *         description: User ID
 *     responses:
 *       200:
 *         description: User retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 user:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     name:
 *                       type: string
 *                     email:
 *                       type: string
 *                     role:
 *                       type: string
 *                       enum: ['owner', 'manager']
 *       404:
 *         description: User not found
 */
router.get('/:userId', protect, getUserById);

/**
 * @swagger
 * /api/v1/users/{userId}:
 *   put:
 *     summary: Update user information
 *     tags: [Users]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         schema:
 *           type: string
 *         required: true
 *         description: User ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *                 description: User name
 *               email:
 *                 type: string
 *                 format: email
 *                 description: User email
 *               password:
 *                 type: string
 *                 description: New password (will be hashed)
 *     responses:
 *       200:
 *         description: User updated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 user:
 *                   type: object
 *                   properties:
 *                     id:
 *                       type: string
 *                     name:
 *                       type: string
 *                     email:
 *                       type: string
 *                     role:
 *                       type: string
 *       403:
 *         description: Not authorized to update this user
 *       404:
 *         description: User not found
 */
router.put('/:userId', protect, updateUser);

/**
 * @swagger
 * /api/v1/users/{userId}:
 *   delete:
 *     summary: Delete user
 *     tags: [Users]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         schema:
 *           type: string
 *         required: true
 *         description: User ID
 *     responses:
 *       200:
 *         description: User deleted successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 deletedUserId:
 *                   type: string
 *       403:
 *         description: Not authorized to delete this user
 *       404:
 *         description: User not found
 */
router.delete('/:userId', protect, deleteUser);

export default router;

