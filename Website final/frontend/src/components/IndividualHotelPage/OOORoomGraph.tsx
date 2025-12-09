import React, { useState, useMemo, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../../redux/store';
import { fetchOOOData, fetchOOOForecasts } from '../../redux/services/api';
import {
    LineChart,
    Line,
    BarChart,
    Bar,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

interface OOORoomGraphProps {
    hotelId: string;
}

interface ChartDataPoint {
    date: string;
    actual?: number;
    predicted?: number;
    isPredicted?: boolean;
}

const OOORoomGraph: React.FC<OOORoomGraphProps> = React.memo(({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');
    const oooRecords = useSelector((state: RootState) => state.records.ooo);
    const oooForecasts = useSelector((state: RootState) => state.forecast.ooo);

    useEffect(() => {
        if (hotelId) {
            dispatch(fetchOOOData(hotelId, timePeriod) as any);
            dispatch(fetchOOOForecasts(hotelId, timePeriod) as any);
        }
    }, [hotelId, timePeriod, dispatch]);

    const { oooData, chartHeight, chartType, xAxisInterval } = useMemo(() => {
        const data: ChartDataPoint[] = [];

        if (oooRecords && oooRecords.length > 0) {
            oooRecords.forEach(record => {
                const formattedDate = new Date(record.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                data.push({
                    date: formattedDate,
                    actual: record.value,
                    isPredicted: false
                });
            });
        }

        if (oooForecasts && oooForecasts.length > 0) {
            oooForecasts.forEach(forecast => {
                const formattedDate = new Date(forecast.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                const existingIndex = data.findIndex(d => d.date === formattedDate);

                if (existingIndex >= 0) {
                    data[existingIndex].predicted = forecast.value;
                } else {
                    data.push({
                        date: formattedDate,
                        predicted: forecast.value,
                        isPredicted: true
                    });
                }
            });
        }

        let chartHeight = 300;
        let chartType: 'bar' | 'line' = 'bar';
        let xAxisInterval = 0;

        switch (timePeriod) {
            case '1w':
                chartHeight = 300;
                chartType = 'bar';
                xAxisInterval = 0;
                break;
            case '1m':
                chartHeight = 350;
                chartType = 'bar';
                xAxisInterval = 2;
                break;
            case '3m':
                chartHeight = 350;
                chartType = 'bar';
                xAxisInterval = 5;
                break;
            case '6m':
                chartHeight = 400;
                chartType = 'line';
                xAxisInterval = 10;
                break;
            case '12m':
                chartHeight = 400;
                chartType = 'line';
                xAxisInterval = 15;
                break;
        }

        return {
            oooData: data,
            chartHeight,
            chartType,
            xAxisInterval
        };
    }, [oooRecords, oooForecasts, timePeriod]);

    return (
        <div className="w-full">
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-6 gap-4">
                    <h3 className="text-lg font-bold text-gray-900 flex items-center">
                        <span className="mr-2">🛠️</span> OOO Room Analysis
                    </h3>
                    <div className="flex flex-wrap gap-2">
                        {(['1w', '1m', '3m', '6m', '12m'] as const).map(period => (
                            <button
                                key={period}
                                className={`
                                    px-3 py-1.5 text-xs font-medium rounded-full transition-all duration-200
                                    ${timePeriod === period
                                        ? 'bg-indigo-600 text-white shadow-md'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                                    }
                                `}
                                onClick={() => setTimePeriod(period)}
                            >
                                {period === '1w' ? '1 Week' : period === '1m' ? '1 Month' : period === '3m' ? '3 Months' : period === '6m' ? '6 Months' : '12 Months'}
                            </button>
                        ))}
                    </div>
                </div>
                <ResponsiveContainer width="100%" height={chartHeight}>
                    {chartType === 'bar' ? (
                        <BarChart data={oooData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                            <XAxis
                                dataKey="date"
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={false}
                                interval={xAxisInterval}
                            />
                            <YAxis
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={false}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#fff',
                                    border: 'none',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                                }}
                                formatter={(value, name) => {
                                    if (name === 'actual') return [value, 'Actual OOO Rooms'];
                                    return [value, 'Predicted OOO Rooms'];
                                }}
                            />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="circle" />
                            <Bar dataKey="actual" fill="#ef4444" name="Actual OOO Rooms" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="predicted" fill="#fca5a5" name="Predicted OOO Rooms" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    ) : (
                        <LineChart data={oooData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                            <XAxis
                                dataKey="date"
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={false}
                                interval={xAxisInterval}
                            />
                            <YAxis
                                tick={{ fill: '#6b7280', fontSize: 12 }}
                                axisLine={{ stroke: '#e5e7eb' }}
                                tickLine={false}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#fff',
                                    border: 'none',
                                    borderRadius: '8px',
                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                                }}
                                formatter={(value, name) => {
                                    if (name === 'actual') return [value, 'Actual OOO Rooms'];
                                    return [value, 'Predicted OOO Rooms'];
                                }}
                            />
                            <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="line" />
                            <Line
                                type="monotone"
                                dataKey="actual"
                                stroke="#ef4444"
                                dot={false}
                                strokeWidth={3}
                                name="Actual OOO Rooms"
                                activeDot={{ r: 6 }}
                            />
                            <Line
                                type="monotone"
                                dataKey="predicted"
                                stroke="#fca5a5"
                                dot={false}
                                strokeWidth={3}
                                strokeDasharray="5 5"
                                name="Predicted OOO Rooms"
                            />
                        </LineChart>
                    )}
                </ResponsiveContainer>
            </div>
    </div>
  );
});

export default OOORoomGraph;