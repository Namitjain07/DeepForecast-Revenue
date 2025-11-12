import { Schema, model, Document } from 'mongoose';

export interface ILastTrain extends Document {
    hotelId: Schema.Types.ObjectId;
    userId: Schema.Types.ObjectId;
    startDateTime: Date;
    endDateTime: Date;
    status: 'none' | 'queued' | 'running' | 'success' | 'failure';
}

const LastTrainSchema = new Schema<ILastTrain>({
    hotelId: {
        type: Schema.Types.ObjectId,
        ref: 'Hotel',
        required: true
    },
    userId: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    startDateTime: {
        type: Date,
        required: true
    },
    endDateTime: {
        type: Date,
        required: true
    },
    status: {
        type: String,
        enum: ['none', 'queued', 'running', 'success', 'failure'],
        default: 'none',
        required: true
    }
}, {
    timestamps: true
});

export default model<ILastTrain>('LastTrain', LastTrainSchema);

