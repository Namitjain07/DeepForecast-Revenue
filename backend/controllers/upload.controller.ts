import { Request, Response } from 'express';
import { uploadToCloudinary, deleteFromCloudinary } from '../config/cloudinary';

export const uploadFile = async (req: Request, res: Response) => {
    try {
        if (!req.body.file) {
            return res.status(400).json({ message: 'No file provided' });
        }

        const fileUrl = await uploadToCloudinary(req.body.file);
        res.status(200).json({
            message: 'File uploaded successfully',
            url: fileUrl
        });
    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ message: 'Error uploading file' });
    }
};

export const deleteFile = async (req: Request, res: Response) => {
    try {
        const { publicId } = req.params;
        await deleteFromCloudinary(publicId);
        res.status(200).json({ message: 'File deleted successfully' });
    } catch (error) {
        console.error('Delete error:', error);
        res.status(500).json({ message: 'Error deleting file' });
    }
};
