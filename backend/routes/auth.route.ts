// @ts-ignore
import express from 'express';
import { login, addUser, addAdmin } from '../controllers/auth.controller';
import { protect, adminOnly } from '../middleware/auth.middleware';

const router = express.Router();

/**
 * @swagger
 * /api/v1/auth/login:
 *   post:
 *     summary: Login for users and admins
 *     tags: [Authentication]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *                 format: email
 *               password:
 *                 type: string
 *     responses:
 *       200:
 *         description: Login successful
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 message:
 *                   type: string
 *                 token:
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
 *       401:
 *         description: Invalid credentials
 */
router.post('/login', login);

/**
 * @swagger
 * /api/v1/auth/add_user:
 *   post:
 *     summary: Add a new user
 *     tags: [Users]
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
 *               - password
 *               - hotelId
 *               - role
 *             properties:
 *               name:
 *                 type: string
 *               email:
 *                 type: string
 *                 format: email
 *               password:
 *                 type: string
 *               hotelId:
 *                 type: string
 *               role:
 *                 type: string
 *                 enum: [owner, manager]
 *     responses:
 *       201:
 *         description: User created successfully
 *       400:
 *         description: Invalid input or email already exists
 *       403:
 *         description: Unauthorized - Insufficient permissions
 */
router.post('/add_user', protect, addUser);

/**
 * @swagger
 * /api/v1/auth/add_admin:
 *   post:
 *     summary: Add a new admin
 *     tags: [Admins]
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
 *               - password
 *             properties:
 *               name:
 *                 type: string
 *               email:
 *                 type: string
 *                 format: email
 *               password:
 *                 type: string
 *     responses:
 *       201:
 *         description: Admin created successfully
 *       400:
 *         description: Invalid input or email already exists
 *       403:
 *         description: Unauthorized - Admin access required
 */
router.post('/add_admin', addAdmin);

export default router;
