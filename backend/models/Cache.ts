import { Schema, model, Document } from 'mongoose';

export interface ICache extends Document {
    hotelId: Schema.Types.ObjectId;
    date: Date;
    revenue: number;
    roomSold: number;
    arrivalRoom: number;
    departureRoom: number;
    oooRoom: number;
}

const CacheSchema = new Schema<ICache>({
    hotelId: {
        type: Schema.Types.ObjectId,
        ref: 'Hotel',
        required: true
    },
    date: {
        type: Date,
        required: true
    },
    revenue: {
        type: Number,
        required: true
    },
    roomSold: {
        type: Number,
        required: true
    },
    arrivalRoom: {
        type: Number,
        required: true
    },
    departureRoom: {
        type: Number,
        required: true
    },
    oooRoom: {
        type: Number,
        required: true
    }
}, {
    timestamps: true
});

export default model<ICache>('Cache', CacheSchema);
