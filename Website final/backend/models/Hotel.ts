import { Schema, model, Document } from 'mongoose';

export interface IHotel extends Document {
    adminId: Schema.Types.ObjectId;
    name: string;
    email: string;
    contactNumber: string;
    plotNo: string;
    streetName: string;
    city: string;
    state: string;
    pincode: string;
    imageUrl?: string;
}

const HotelSchema = new Schema<IHotel>({
    adminId: {
        type: Schema.Types.ObjectId,
        ref: 'Admin',
        required: true
    },
    name: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true
    },
    contactNumber: {
        type: String,
        required: true
    },
    plotNo: {
        type: String,
        required: true
    },
    streetName: {
        type: String,
        required: true
    },
    city: {
        type: String,
        required: true
    },
    state: {
        type: String,
        required: true
    },
    pincode: {
        type: String,
        required: true
    },
    imageUrl: {
        type: String,
        default: null
    }
}, {
    timestamps: true
});

// Index for faster queries by admin
HotelSchema.index({ adminId: 1 });

export default model<IHotel>('Hotel', HotelSchema);
