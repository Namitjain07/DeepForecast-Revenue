import { Schema, model, Document } from 'mongoose';

export interface IRecord extends Document {
    hotelId: Schema.Types.ObjectId;
    date: Date;
    roomsSold: number;
    day: string;
    arrivalRooms: number;
    complimentRooms: number;
    houseUse: number;
    individualConfirm: number;
    occupancyPercentage: number;
    roomRevenue: number;
    averageRoomRate: number;
    departureRooms: number;
    oooRooms: number;
    pax: number;
    totalRoomInventory: number;
}

const RecordSchema = new Schema<IRecord>({
    hotelId: {
        type: Schema.Types.ObjectId,
        ref: 'Hotel',
        required: true
    },
    date: {
        type: Date,
        required: true
    },
    roomsSold: {
        type: Number,
        required: true
    },
    day: {
        type: String,
        required: true
    },
    arrivalRooms: {
        type: Number,
        required: true
    },
    complimentRooms: {
        type: Number,
        required: true
    },
    houseUse: {
        type: Number,
        required: true
    },
    individualConfirm: {
        type: Number,
        required: true
    },
    occupancyPercentage: {
        type: Number,
        required: true
    },
    roomRevenue: {
        type: Number,
        required: true
    },
    averageRoomRate: {
        type: Number,
        required: true
    },
    departureRooms: {
        type: Number,
        required: true
    },
    oooRooms: {
        type: Number,
        required: true
    },
    pax: {
        type: Number,
        required: true
    },
    totalRoomInventory: {
        type: Number,
        required: true
    }
}, {
    timestamps: true
});

export default model<IRecord>('Record', RecordSchema);
