import React, { useState, useMemo } from 'react';
import {
    BarChart,
    Bar,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import '../../stylesheet/ui/component-ui-departure-room.css';

interface DepartureRoomGraphProps {
    hotelId: string;
}

const DepartureRoomGraph: React.FC<DepartureRoomGraphProps> = ({ hotelId }) => {
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');

    const departureData = useMemo(() => {
        const daysMap = {
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '12m': 365
        };

        const days = daysMap[timePeriod];
        const data = [];
        const baseDate = new Date(2025, 0, 1);

        for (let i = 0; i < days; i++) {
            const date = new Date(baseDate);
            date.setDate(date.getDate() + i);
            const formattedDate = date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric'
            });

            data.push({
                date: formattedDate,
                actual: Math.floor(Math.random() * 60) + 15,
                predicted: Math.floor(Math.random() * 65) + 10
            });
        }

        return data;
    }, [timePeriod]);

    return (
        <div className="component-ui-departure-room-container">
            <div className="component-ui-departure-room-header">
                <h3 className="component-ui-departure-room-title">Departure Room Analysis</h3>
                <div className="component-ui-departure-room-toggle">
                    {(['1w', '1m', '3m', '6m', '12m'] as const).map(period => (
                        <button
                            key={period}
                            className={`component-ui-departure-room-toggle-btn ${timePeriod === period ? 'active' : ''}`}
                            onClick={() => setTimePeriod(period)}
                        >
                            {period === '1w' ? '1 Week' : period === '1m' ? '1 Month' : period === '3m' ? '3 Months' : period === '6m' ? '6 Months' : '12 Months'}
                        </button>
                    ))}
                </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart
                    data={departureData}
                    margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: '#666', fontSize: 12 }}
                        interval={Math.floor(departureData.length / 8)}
                    />
                    <YAxis tick={{ fill: '#666', fontSize: 12 }} />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e0e0e0',
                            borderRadius: '4px',
                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                        }}
                        formatter={(value) => value}
                    />
                    <Legend
                        wrapperStyle={{ paddingTop: '20px' }}
                        iconType="circle"
                    />
                    <Bar dataKey="actual" fill="#ffa500" name="Actual Departures" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="predicted" fill="#2ecc71" name="Predicted Departures" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default DepartureRoomGraph;

