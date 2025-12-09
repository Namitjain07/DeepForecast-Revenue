// Seeding script: connects to MongoDB (MONGO_URI from .env), clears collections and inserts 10 fake entries for Admin, Hotel, User, Record and Cache.

import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
// import * as crypto from 'crypto';

dotenv.config();

import Admin from './models/Admin';
import Hotel from './models/Hotel';
import User from './models/User';
import Record from './models/Record';
import Forecast from './models/Forecast';
import LastTrain from './models/Lasttrain';

const MONGO_URI = process.env.MONGODB_URI || process.env.MONGODB_URI || '';

if (!MONGO_URI) {
    console.error('MONGO_URI not set in .env');
    process.exit(1);
}

import * as bcrypt from 'bcryptjs';

async function hashPassword (password: string) {
    const salt = await bcrypt.genSalt(10);
    return await bcrypt.hash(password, salt);
}


function randomInt(min: number, max: number) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

async function seed() {
    try {
        await mongoose.connect(MONGO_URI, {
            // connection options left default
        } as any);
        console.log('Connected to MongoDB for seeding');

        // Drop old indexes from previous schema versions to avoid conflicts
        try {
            const adminCollection = mongoose.connection.collection('admins');
            const adminIndexes = await adminCollection.getIndexes();
            for (const indexName of Object.keys(adminIndexes)) {
                if (indexName !== '_id_' && indexName.includes('AID')) {
                    await adminCollection.dropIndex(indexName);
                    console.log(`Dropped old index: ${indexName}`);
                }
            }

            const userCollection = mongoose.connection.collection('users');
            const userIndexes = await userCollection.getIndexes();
            for (const indexName of Object.keys(userIndexes)) {
                if (indexName !== '_id_' && indexName.includes('UID')) {
                    await userCollection.dropIndex(indexName);
                    console.log(`Dropped old index: ${indexName}`);
                }
            }

            const hotelCollection = mongoose.connection.collection('hotels');
            const hotelIndexes = await hotelCollection.getIndexes();
            for (const indexName of Object.keys(hotelIndexes)) {
                if (indexName !== '_id_' && indexName.includes('HID')) {
                    await hotelCollection.dropIndex(indexName);
                    console.log(`Dropped old index: ${indexName}`);
                }
            }

            const recordCollection = mongoose.connection.collection('records');
            const recordIndexes = await recordCollection.getIndexes();
            for (const indexName of Object.keys(recordIndexes)) {
                if (indexName !== '_id_' && indexName.includes('RID')) {
                    await recordCollection.dropIndex(indexName);
                    console.log(`Dropped old index: ${indexName}`);
                }
            }

            const cacheCollection = mongoose.connection.collection('caches');
            const cacheIndexes = await cacheCollection.getIndexes();
            for (const indexName of Object.keys(cacheIndexes)) {
                if (indexName !== '_id_' && indexName.includes('CID')) {
                    await cacheCollection.dropIndex(indexName);
                    console.log(`Dropped old index: ${indexName}`);
                }
            }
        } catch (indexErr) {
            console.log('Index cleanup (optional):', (indexErr as any).message);
        }

        // Clear existing data in collections
        await Admin.deleteMany({});
        await Hotel.deleteMany({});
        await User.deleteMany({});
        await Record.deleteMany({});
        await Forecast.deleteMany({});
        await LastTrain.deleteMany({});

        // Create 10 Admins
        const adminDocs: any[] = [];
        for (let i = 1; i <= 10; i++) {
            adminDocs.push({
                name: `Admin ${i}`,
                email: `admin${i}@example.com`,
                password: await hashPassword('password123'),
                imageUrl: `https://picsum.photos/seed/admin${i}/200/200`
            });
        }
        const savedAdmins = (await Admin.insertMany(adminDocs)) as any[];
        console.log(`Inserted ${savedAdmins.length} admins`);
        if (!savedAdmins || savedAdmins.length === 0) {
            throw new Error('No admins were created during seeding');
        }

        // Create 10 Hotels (assign to admins round-robin)
        const hotelDocs: any[] = [];
        for (let i = 1; i <= 10; i++) {
            // safe because savedAdmins checked above
            const adminForHotel = savedAdmins[(i - 1) % savedAdmins.length] as any;
            hotelDocs.push({
                adminId: adminForHotel._id,
                name: `Hotel ${i}`,
                email: `hotel${i}@example.com`,
                contactNumber: `${randomInt(9000000000, 9999999999)}`,
                plotNo: `${i}A`,
                streetName: `Street ${i}`,
                city: `City ${i}`,
                state: `State ${i}`,
                pincode: `${100000 + i}`,
                imageUrl: `https://picsum.photos/seed/hotel${i}/400/300`
            });
        }
        const savedHotels = (await Hotel.insertMany(hotelDocs)) as any[];
        console.log(`Inserted ${savedHotels.length} hotels`);
        if (!savedHotels || savedHotels.length === 0) {
            throw new Error('No hotels were created during seeding');
        }

        // Create 10 Users (assign to hotels: 1 owner + 1 manager per hotel)
        const userDocs: any[] = [];
        for (let i = 1; i <= 10; i++) {
            const hotelForUser = savedHotels[(i - 1) % savedHotels.length] as any;

            // Create owner for this hotel
            userDocs.push({
                hotelId: hotelForUser._id,
                name: `Owner ${i}`,
                email: `owner${i}@example.com`,
                password: await hashPassword('ownerpass123'),
                role: 'owner',
                imageUrl: `https://picsum.photos/seed/owner${i}/200/200`
            });

            // Create manager for this hotel
            userDocs.push({
                hotelId: hotelForUser._id,
                name: `Manager ${i}`,
                email: `manager${i}@example.com`,
                password: await hashPassword('managerpass123'),
                role: 'manager',
                imageUrl: `https://picsum.photos/seed/manager${i}/200/200`
            });
        }
        const savedUsers = (await User.insertMany(userDocs)) as any[];
        console.log(`Inserted ${savedUsers.length} users (owners and managers)`);

        // Create time series Records (one entry per day for 1 year back from today per hotel)
        const recordDocs: any[] = [];
        const today = new Date();
        const recordStartDate = new Date(today);
        recordStartDate.setDate(recordStartDate.getDate() - 365); // 1 year back
        const recordEndDate = new Date(today);

        // Calculate number of days exactly to be safe across DST boundaries
        const msPerDay = 24 * 60 * 60 * 1000;
        const recordDaysCount = Math.round((recordEndDate.getTime() - recordStartDate.getTime()) / msPerDay) + 1;

        for (let hotelIdx = 0; hotelIdx < savedHotels.length; hotelIdx++) {
            const hotel = savedHotels[hotelIdx] as any;
            // Generate records for each day in the past 365 days
            for (let dayOffset = 0; dayOffset < recordDaysCount; dayOffset++) {
                const date = new Date(recordStartDate);
                date.setDate(date.getDate() + dayOffset);

                const roomsSold = randomInt(20, 200);
                const totalInventory = randomInt(100, 250);
                const occupancy = Math.round((roomsSold / totalInventory) * 100 * 100) / 100;

                recordDocs.push({
                    hotelId: hotel._id,
                    date,
                    roomsSold,
                    day: date.toLocaleString('en-US', { weekday: 'long' }),
                    arrivalRooms: randomInt(50, 150),
                    complimentRooms: randomInt(0, 5),
                    houseUse: randomInt(0, 5),
                    individualConfirm: randomInt(0, 10),
                    occupancyPercentage: occupancy,
                    roomRevenue: Math.round(roomsSold * randomInt(100, 500)),
                    averageRoomRate: Math.round((randomInt(100, 500) + Math.random()) * 100) / 100,
                    departureRooms: randomInt(10, 100),
                    oooRooms: randomInt(0, 3),
                    pax: roomsSold * randomInt(1, 3),
                    totalRoomInventory: totalInventory
                });
            }
        }
        const savedRecords = (await Record.insertMany(recordDocs)) as any[];
        console.log(`Inserted ${savedRecords.length} time series records (${recordDaysCount} days × ${savedHotels.length} hotels for 1 year back from today)`);

        // Create time series Forecast data (one entry per day for 1 year forward from today per hotel)
        const forecastDocs: any[] = [];
        const forecastStartDate = new Date(today);
        forecastStartDate.setDate(forecastStartDate.getDate() + 1); // Start tomorrow
        const forecastDays = 365; // 1 year forward

        for (let hotelIdx = 0; hotelIdx < savedHotels.length; hotelIdx++) {
            const hotel = savedHotels[hotelIdx] as any;
            // Generate forecast for 365 days into the future
            for (let dayOffset = 0; dayOffset < forecastDays; dayOffset++) {
                const date = new Date(forecastStartDate);
                date.setDate(date.getDate() + dayOffset);

                const roomSold = randomInt(20, 200);
                const revenue = Math.round(roomSold * randomInt(100, 500));

                forecastDocs.push({
                    hotelId: hotel._id,
                    date,
                    revenue,
                    roomSold,
                    arrivalRoom: randomInt(30, 120),
                    departureRoom: randomInt(10, 80),
                    oooRoom: randomInt(0, 5)
                });
            }
        }
        const savedForecasts = (await Forecast.insertMany(forecastDocs)) as any[];
        console.log(`Inserted ${savedForecasts.length} time series forecast entries (${forecastDays} days × ${savedHotels.length} hotels for 1 year forward from today)`);

        // Create 10 LastTrain entries (one per hotel)
        const lastTrainDocs: any[] = [];
        const statuses: Array<'none' | 'queued' | 'running' | 'success' | 'failure'> = ['none', 'queued', 'running', 'success', 'failure'];
        for (let i = 0; i < 10; i++) {
            const hotel = savedHotels[i] as any;
            const user = savedUsers[i * 2] as any; // Use owner for last train
            const startDateTime = new Date();
            startDateTime.setDate(startDateTime.getDate() - randomInt(1, 7));
            startDateTime.setHours(randomInt(0, 23), randomInt(0, 59), 0);

            const endDateTime = new Date(startDateTime);
            endDateTime.setMinutes(endDateTime.getMinutes() + randomInt(5, 120));

            const status = statuses[randomInt(0, statuses.length - 1)];

            lastTrainDocs.push({
                hotelId: hotel._id,
                userId: user._id,
                startDateTime,
                endDateTime,
                status
            });
        }
        const savedLastTrains = (await LastTrain.insertMany(lastTrainDocs)) as any[];
        console.log(`Inserted ${savedLastTrains.length} last train entries`);

        console.log('Seeding completed successfully');
    } catch (err) {
        console.error('Seeding failed', err);
        process.exitCode = 1;
    } finally {
        await mongoose.disconnect();
        process.exit(0);
    }
}

seed();
