// @ts-ignore
import express from 'express';
import { uploadFile, deleteFile } from '../controllers/upload.controller';

const router = express.Router();

/**
 * @swagger
 * /api/v1/upload:
 *   post:
 *     summary: Upload a file to Cloudinary
 *     security:
 *       - bearerAuth: []
 *     tags: [Upload]
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *     responses:
 *       200:
 *         description: File uploaded successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Server error
 */
router.post('/', uploadFile);

/**
 * @swagger
 * /api/v1/upload/{publicId}:
 *   delete:
 *     summary: Delete a file from Cloudinary
 *     security:
 *       - bearerAuth: []
 *     tags: [Upload]
 *     parameters:
 *       - in: path
 *         name: publicId
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: File deleted successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Server error
 */
router.delete('/:publicId', deleteFile);

export default router;
