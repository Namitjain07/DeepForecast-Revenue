// Seeding script: connects to MongoDB (MONGO_URI from .env), clears collections and inserts 10 fake entries for Admin, Hotel, User, Record and Cache.

import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
// import * as crypto from 'crypto';

dotenv.config();

import Admin from './models/Admin';
import Hotel from './models/Hotel';
import User from './models/User';
import Record from './models/Record';
import Cache from './models/Cache';

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

function randomDateWithinDays(daysBack: number) {
    const now = new Date();
    const past = new Date(now.getTime() - Math.floor(Math.random() * daysBack) * 24 * 60 * 60 * 1000);
    return past;
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
        await Cache.deleteMany({});

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

        // Create 10 Users (assign to hotels round-robin) with roles owner/manager
        const userDocs: any[] = [];
        const roles: Array<'owner' | 'manager'> = ['owner', 'manager'];
        for (let i = 1; i <= 10; i++) {
            // safe because savedHotels checked above
            const hotelForUser = savedHotels[(i - 1) % savedHotels.length] as any;
            const role = roles[i % 2];
            userDocs.push({
                hotelId: hotelForUser._id,
                name: `User ${i}`,
                email: `user${i}@example.com`,
                password: await hashPassword('userpass123'),
                role,
                imageUrl: `https://picsum.photos/seed/user${i}/200/200`
            });
        }
        const savedUsers = (await User.insertMany(userDocs)) as any[];
        console.log(`Inserted ${savedUsers.length} users`);

        // Create 10 Records (assign randomly to hotels)
        const recordDocs: any[] = [];
        for (let i = 1; i <= 1000; i++) {
            const hotel = savedHotels[randomInt(0, savedHotels.length - 1)] as any; // safe due to earlier check
            const date = randomDateWithinDays(30);
            const roomsSold = randomInt(20, 200);
            const totalInventory = randomInt(100, 250);
            const occupancy = Math.round((roomsSold / totalInventory) * 100 * 100) / 100; // percent
            const arr = randomInt(50, 300);
            const depart = randomInt(10, 100);

            recordDocs.push({
                hotelId: hotel._id,
                date,
                roomsSold,
                day: date.toLocaleString('en-US', { weekday: 'long' }),
                arrivalRooms: arr,
                complimentRooms: randomInt(0, 5),
                houseUse: randomInt(0, 5),
                individualConfirm: randomInt(0, 10),
                occupancyPercentage: occupancy,
                roomRevenue: randomInt(2000, 50000),
                averageRoomRate: Math.round((randomInt(100, 500) + Math.random()) * 100) / 100,
                departureRooms: depart,
                oooRooms: randomInt(0, 3),
                pax: randomInt(20, 300),
                totalRoomInventory: totalInventory,
                snapshotDate: new Date(),
                arrivalDate: date,
                actualOrForecast: Math.random() > 0.5 ? 'actual' : 'forecast'
            });
        }
        const savedRecords = (await Record.insertMany(recordDocs)) as any[];
        console.log(`Inserted ${savedRecords.length} records`);

        // Create 10 Cache entries
        const cacheDocs: any[] = [];
        for (let i = 1; i <= 100; i++) {
            const hotel = savedHotels[randomInt(0, savedHotels.length - 1)] as any; // safe due to earlier check
            const date = randomDateWithinDays(10);
            const roomSold = randomInt(20, 200);
            const revenue = Math.round(roomSold * randomInt(100, 500));

            cacheDocs.push({
                hotelId: hotel._id,
                date,
                revenue,
                roomSold,
                arrivalRoom: randomInt(0, 50),
                departureRoom: randomInt(0, 50),
                oooRoom: randomInt(0, 5)
            });
        }
        const savedCaches = (await Cache.insertMany(cacheDocs)) as any[];
        console.log(`Inserted ${savedCaches.length} cache entries`);

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
