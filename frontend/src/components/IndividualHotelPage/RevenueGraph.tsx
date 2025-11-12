import React, { useState, useMemo } from 'react';
import {
    AreaChart,
    Area,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import '../../stylesheet/ui/component-ui-revenue-graph.css';

interface RevenueGraphProps {
    hotelId: string;
}

interface RevenueData {
    date: string;
    actual: number;
    predicted: number;
}

const RevenueGraph: React.FC<RevenueGraphProps> = ({ hotelId }) => {
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');

    const revenueData = useMemo(() => {
        const daysMap = {
            '1w': 7,
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '12m': 365
        };

        const days = daysMap[timePeriod];
        const data: RevenueData[] = [];
        let base = +new Date(2025, 0, 1);
        let oneDay = 24 * 3600 * 1000;
        let actualData = Math.random() * 10000;
        let predictedData = Math.random() * 9000;

        for (let i = 0; i < days; i++) {
            const now = new Date((base += oneDay));
            const dateStr = [now.getFullYear(), now.getMonth() + 1, now.getDate()].join('/');

            actualData = Math.round((Math.random() - 0.5) * 2000 + actualData);
            predictedData = Math.round((Math.random() - 0.5) * 1800 + predictedData);

            data.push({
                date: dateStr,
                actual: Math.max(actualData, 0),
                predicted: Math.max(predictedData, 0)
            });
        }

        return data;
    }, [timePeriod]);

    return (
        <div className="component-ui-departure-room-container">
            <div className="component-ui-departure-room-header">
                <h3 className="component-ui-departure-room-title">Revenue Analysis</h3>
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
            <ResponsiveContainer width="100%" height={400}>
                <AreaChart
                    data={revenueData}
                    margin={{top: 20, right: 30, left: 0, bottom: 20}}
                >
                    <defs>
                        <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="rgba(255, 158, 68, 0.8)" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="rgba(255, 99, 132, 0.8)" stopOpacity={0.1}/>
                        </linearGradient>
                        <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="rgba(100, 200, 255, 0.6)" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="rgba(54, 162, 235, 0.6)" stopOpacity={0.1}/>
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0"/>
                    <XAxis
                        dataKey="date"
                        tick={{fill: '#666', fontSize: 12}}
                        interval={Math.floor(revenueData.length / 10)}
                    />
                    <YAxis
                        tick={{fill: '#666', fontSize: 12}}
                        label={{value: 'Revenue ($)', angle: -90, position: 'insideLeft'}}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e0e0e0',
                            borderRadius: '4px',
                            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                        }}
                        formatter={(value) => `$${value}`}
                        labelFormatter={(label) => `Date: ${label}`}
                    />
                    <Legend
                        wrapperStyle={{paddingTop: '20px'}}
                        iconType="line"
                    />
                    <Area
                        type="monotone"
                        dataKey="actual"
                        stroke="rgb(255, 99, 132)"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorActual)"
                        name="Actual Revenue"
                        isAnimationActive={true}
                    />
                    <Area
                        type="monotone"
                        dataKey="predicted"
                        stroke="rgb(54, 162, 235)"
                        strokeWidth={2}
                        fillOpacity={1}
                        fill="url(#colorPredicted)"
                        name="Predicted Revenue"
                        isAnimationActive={true}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default RevenueGraph;
