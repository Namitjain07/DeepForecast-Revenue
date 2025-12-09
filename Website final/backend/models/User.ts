import { Schema, model, Document } from 'mongoose';

export interface IUser extends Document {
    hotelId: Schema.Types.ObjectId;
    name: string;
    email: string;
    password: string;
    role: 'owner' | 'manager';
    imageUrl?: string;
}

const UserSchema = new Schema<IUser>({
    hotelId: {
        type: Schema.Types.ObjectId,
        ref: 'Hotel',
        required: true
    },
    name: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true,
        unique: true
    },
    password: {
        type: String,
        required: true
    },
    role: {
        type: String,
        enum: ['owner', 'manager'],
        required: true
    },
    imageUrl: {
        type: String,
        default: null
    }
}, {
    timestamps: true
});

// Index for faster queries by hotel
UserSchema.index({ hotelId: 1 });

export default model<IUser>('User', UserSchema);
