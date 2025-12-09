import { Schema, model, Document } from 'mongoose';

export interface IForecast extends Document {
    hotelId: Schema.Types.ObjectId;
    date: Date;
    revenue: number;
    roomSold: number;
    arrivalRoom: number;
    departureRoom: number;
    oooRoom: number;
}

const ForecastSchema = new Schema<IForecast>({
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

// Add compound index for faster queries by hotel and date
ForecastSchema.index({ hotelId: 1, date: -1 });

export default model<IForecast>('Forecast', ForecastSchema);

