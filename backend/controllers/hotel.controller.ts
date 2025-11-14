import { Request, Response } from 'express';
import Hotel from '../models/Hotel';
import User from '../models/User';
import * as bcrypt from 'bcryptjs';

class ApiError extends Error {
    statusCode: number;
    constructor(message: string, statusCode: number) {
        super(message);
        this.statusCode = statusCode;
    }
}

export const addHotel = async (req: Request, res: Response) => {
    try {
        const {
            name,
            email,
            contactNumber,
            plotNo,
            streetName,
            city,
            state,
            pincode,
            imageUrl,
            ownerName,
            ownerEmail,
            ownerPassword
        } = req.body;

        // Check for required hotel fields
        const requiredHotelFields = [
            'name',
            'email',
            'contactNumber',
            'plotNo',
            'streetName',
            'city',
            'state',
            'pincode'
        ];

        const missingHotelFields = requiredHotelFields.filter(field => !req.body[field]);
        if (missingHotelFields.length > 0) {
            throw new ApiError(`Missing required hotel fields: ${missingHotelFields.join(', ')}`, 400);
        }

        // Check for required owner fields
        if (!ownerName || !ownerEmail || !ownerPassword) {
            throw new ApiError('Missing required owner fields: ownerName, ownerEmail, ownerPassword', 400);
        }

        // Get admin ID from authenticated user
        // @ts-ignore
        const adminId = req.user._id;

        // Check if hotel with same email exists
        const hotelExists = await Hotel.findOne({ email });
        if (hotelExists) {
            throw new ApiError('Hotel with this email already exists', 400);
        }

        // Check if owner email already exists
        const ownerExists = await User.findOne({ email: ownerEmail });
        if (ownerExists) {
            throw new ApiError('Owner with this email already exists', 400);
        }

        // Hash owner password
        const salt = await bcrypt.genSalt(10);
        const hashedPassword = await bcrypt.hash(ownerPassword, salt);

        // Create hotel
        const hotel = await Hotel.create({
            adminId,
            name,
            email,
            contactNumber,
            plotNo,
            streetName,
            city,
            state,
            pincode,
            imageUrl
        });

        // Create owner user for the hotel
        const owner = await User.create({
            hotelId: hotel._id,
            name: ownerName,
            email: ownerEmail,
            password: hashedPassword,
            role: 'owner',
            imageUrl: imageUrl || `https://picsum.photos/seed/owner${Date.now()}/200/200`
        });

        res.status(201).json({
            message: 'Hotel and owner created successfully',
            hotel: {
                id: hotel._id,
                name: hotel.name,
                email: hotel.email,
                city: hotel.city,
                state: hotel.state,
                imageUrl: hotel.imageUrl,
                owner: {
                    id: owner._id,
                    name: owner.name,
                    email: owner.email,
                    role: owner.role
                }
            }
        });

    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const getRecentlyAddedHotels = async (req: Request, res: Response) => {
    try {
        const { limit = 5 } = req.body;

        // Validate limit
        const parsedLimit = Math.min(parseInt(limit) || 5, 100);

        // Fetch recently added hotels (sorted by creation date, newest first)
        const hotels = await Hotel.find()
            .sort({ createdAt: -1 })
            .limit(parsedLimit)
            .select('name email contactNumber city imageUrl _id createdAt');

        // Format response with ownerName from User table
        const formattedHotels = await Promise.all(
            hotels.map(async (hotel: any) => {
                // Find owner user for this hotel
                const owner = await User.findOne(
                    { hotelId: hotel._id, role: 'owner' },
                    'name'
                );

                return {
                    id: hotel._id,
                    hotelName: hotel.name,
                    ownerName: owner?.name || 'N/A',
                    city: hotel.city,
                    contactNumber: hotel.contactNumber,
                    imageUrl: hotel.imageUrl,
                    addedAt: hotel.createdAt
                };
            })
        );

        res.status(200).json({
            message: 'Recently added hotels retrieved successfully',
            count: formattedHotels.length,
            hotels: formattedHotels
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const searchHotels = async (req: Request, res: Response) => {
    try {
        const { searchTerm, page = 1, limit = 10 } = req.body;

        if (!searchTerm || searchTerm.trim() === '') {
            throw new ApiError('Search term is required', 400);
        }

        const pageNum = Math.max(1, parseInt(page) || 1);
        const limitNum = Math.min(parseInt(limit) || 10, 100);
        const skip = (pageNum - 1) * limitNum;

        // First, find all users (owners) matching the search term
        const matchingOwners = await User.find(
            {
                $and: [
                    { role: 'owner' },
                    { name: { $regex: searchTerm, $options: 'i' } }
                ]
            },
            'hotelId'
        );

        // Extract hotel IDs from matching owners
        const ownerHotelIds = matchingOwners.map(owner => owner.hotelId);

        // Create search query - find hotels by name OR by owner
        const searchQuery = {
            $or: [
                { name: { $regex: searchTerm, $options: 'i' } },
                { _id: { $in: ownerHotelIds } }
            ]
        };

        // Get total count for pagination
        const totalCount = await Hotel.countDocuments(searchQuery);

        // Fetch hotels with pagination
        const hotels = await Hotel.find(searchQuery)
            .limit(limitNum)
            .skip(skip)
            .select('name email contactNumber city imageUrl _id createdAt');

        // Format response with ownerName and managerName from User table
        const formattedHotels = await Promise.all(
            hotels.map(async (hotel: any) => {
                // Find owner user for this hotel
                const owner = await User.findOne(
                    { hotelId: hotel._id, role: 'owner' },
                    'name'
                );

                // Find manager user for this hotel
                const manager = await User.findOne(
                    { hotelId: hotel._id, role: 'manager' },
                    'name'
                );

                return {
                    id: hotel._id,
                    name: hotel.name,
                    ownerName: owner?.name || 'N/A',
                    // managerName: manager?.name || 'N/A',
                    city: hotel.city,
                    contactNumber: hotel.contactNumber,
                    imageUrl: hotel.imageUrl
                };
            })
        );

        res.status(200).json({
            message: 'Hotels found',
            searchTerm,
            pagination: {
                currentPage: pageNum,
                pageSize: limitNum,
                totalCount,
                totalPages: Math.ceil(totalCount / limitNum),
                hasNextPage: skip + limitNum < totalCount
            },
            hotels: formattedHotels
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const getAllHotels = async (req: Request, res: Response) => {
    try {
        const { page = 1, limit = 10 } = req.query;

        const pageNum = Math.max(1, parseInt(page as string) || 1);
        const limitNum = Math.min(parseInt(limit as string) || 10, 100);
        const skip = (pageNum - 1) * limitNum;

        // Get total count of hotels
        const totalCount = await Hotel.countDocuments();

        // Fetch hotels with pagination
        const hotels = await Hotel.find()
            .sort({ createdAt: -1 })
            .limit(limitNum)
            .skip(skip)
            .select('name email contactNumber city state imageUrl _id createdAt');

        // Format response with ownerName from User table
        const formattedHotels = await Promise.all(
            hotels.map(async (hotel: any) => {
                // Find owner user for this hotel
                const owner = await User.findOne(
                    { hotelId: hotel._id, role: 'owner' },
                    'name'
                );

                return {
                    id: hotel._id,
                    name: hotel.name,
                    email: hotel.email,
                    contactNumber: hotel.contactNumber,
                    city: hotel.city,
                    state: hotel.state,
                    imageUrl: hotel.imageUrl,
                    ownerName: owner?.name || 'N/A'
                };
            })
        );

        res.status(200).json({
            message: 'Hotels retrieved successfully',
            pagination: {
                currentPage: pageNum,
                pageSize: limitNum,
                totalCount,
                totalPages: Math.ceil(totalCount / limitNum),
                hasNextPage: skip + limitNum < totalCount,
                hasPreviousPage: pageNum > 1
            },
            hotels: formattedHotels
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get hotel general information
 */
export const getHotelGeneralInfo = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        const hotel = await Hotel.findById(hotelId).select(
            'name email contactNumber plotNo streetName city state pincode'
        );

        if (!hotel) {
            throw new ApiError('Hotel not found', 404);
        }

        res.status(200).json({
            message: 'Hotel general information retrieved successfully',
            hotel: {
                id: hotel._id,
                name: hotel.name,
                email: hotel.email,
                contactNumber: hotel.contactNumber,
                plotNo: hotel.plotNo,
                streetName: hotel.streetName,
                city: hotel.city,
                state: hotel.state,
                pincode: hotel.pincode
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

/**
 * Get dashboard stats for a hotel (Total Revenue, Rooms Sold, Avg Occupancy for last 30 days)
 */
export const getDashboardStats = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Import Record model
        const Record = require('../models/Record').default || require('../models/Record');

        // Calculate last 30 days date range
        const today = new Date();
        const thirtyDaysAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

        // Get records for last 30 days
        const records = await Record.find({
            hotelId,
            date: { $gte: thirtyDaysAgo, $lte: today }
        }).select('roomRevenue roomsSold occupancyPercentage');

        if (!records || records.length === 0) {
            return res.status(200).json({
                message: 'Dashboard stats retrieved successfully',
                stats: {
                    totalRevenue: 0,
                    totalRoomsSold: 0,
                    avgOccupancyRate: 0,
                    period: 'Last 30 days'
                }
            });
        }

        // Calculate stats
        const totalRevenue = records.reduce((sum: number, record: any) => sum + (record.roomRevenue || 0), 0);
        const totalRoomsSold = records.reduce((sum: number, record: any) => sum + (record.roomsSold || 0), 0);
        const avgOccupancyRate = records.length > 0
            ? records.reduce((sum: number, record: any) => sum + (record.occupancyPercentage || 0), 0) / records.length
            : 0;

        res.status(200).json({
            message: 'Dashboard stats retrieved successfully',
            stats: {
                totalRevenue: Math.round(totalRevenue),
                totalRoomsSold,
                avgOccupancyRate: parseFloat(avgOccupancyRate.toFixed(2)),
                period: 'Last 30 days',
                recordsCount: records.length
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};

export const updateHotelInfo = async (req: Request, res: Response) => {
    try {
        const { hotelId } = req.params;
        const { name, email, contactNumber, plotNo, streetName, city, state, pincode } = req.body;

        if (!hotelId) {
            throw new ApiError('Hotel ID is required', 400);
        }

        // Validate required fields
        if (!name || !email || !contactNumber || !plotNo || !streetName || !city || !state || !pincode) {
            throw new ApiError('All fields are required', 400);
        }

        const updatedHotel = await Hotel.findByIdAndUpdate(
            hotelId,
            {
                name,
                email,
                contactNumber,
                plotNo,
                streetName,
                city,
                state,
                pincode
            },
            { new: true, runValidators: true }
        ).select('name email contactNumber plotNo streetName city state pincode');

        if (!updatedHotel) {
            throw new ApiError('Hotel not found', 404);
        }

        res.status(200).json({
            message: 'Hotel information updated successfully',
            hotel: {
                id: updatedHotel._id,
                name: updatedHotel.name,
                email: updatedHotel.email,
                contactNumber: updatedHotel.contactNumber,
                plotNo: updatedHotel.plotNo,
                streetName: updatedHotel.streetName,
                city: updatedHotel.city,
                state: updatedHotel.state,
                pincode: updatedHotel.pincode
            }
        });
    } catch (error: any) {
        const statusCode = error.statusCode || 500;
        res.status(statusCode).json({
            message: error.message || 'An unexpected error occurred'
        });
    }
};