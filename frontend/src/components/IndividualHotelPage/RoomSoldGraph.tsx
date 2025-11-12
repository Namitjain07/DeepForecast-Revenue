import React, { useState, useMemo } from 'react';
import {
    BarChart,
    Bar,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer,
    Cell
} from 'recharts';
import '../../stylesheet/ui/component-ui-room-sold-graph.css';

interface RoomSoldGraphProps {
    hotelId: string;
}

const RoomSoldGraph: React.FC<RoomSoldGraphProps> = ({ hotelId }) => {
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');

    const roomsData = useMemo(() => {
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
                actual: Math.floor(Math.random() * 80) + 20,
                predicted: Math.floor(Math.random() * 85) + 15
            });
        }

        return data;
    }, [timePeriod]);

    return (
        <div className="component-ui-room-sold-graph-container">
            <div className="component-ui-room-sold-graph-header">
                <h3 className="component-ui-room-sold-graph-title">Rooms Sold Analysis</h3>
                <div className="component-ui-room-sold-graph-toggle">
                    {(['1w', '1m', '3m', '6m', '12m'] as const).map(period => (
                        <button
                            key={period}
                            className={`component-ui-room-sold-graph-toggle-btn ${timePeriod === period ? 'active' : ''}`}
                            onClick={() => setTimePeriod(period)}
                        >
                            {period === '1w' ? '1 Week' : period === '1m' ? '1 Month' : period === '3m' ? '3 Months' : period === '6m' ? '6 Months' : '12 Months'}
                        </button>
                    ))}
                </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
                <BarChart
                    data={roomsData}
                    margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
                >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis
                        dataKey="date"
                        tick={{ fill: '#666', fontSize: 12 }}
                        interval={Math.floor(roomsData.length / 8)}
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
                    <Bar dataKey="actual" fill="#ff6384" name="Actual Rooms" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="predicted" fill="#36a2eb" name="Predicted Rooms" radius={[4, 4, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default RoomSoldGraph;

