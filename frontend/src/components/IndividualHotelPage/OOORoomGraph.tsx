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
import '../../stylesheet/ui/component-ui-ooo-room-graph.css';

interface OOORoomGraphProps {
    hotelId: string;
}

interface ChartDataPoint {
    date: string;
    actual?: number;
    predicted?: number;
    isPredicted?: boolean;
}

const OOORoomGraph: React.FC<OOORoomGraphProps> = ({ hotelId }) => {
    const dispatch = useDispatch<AppDispatch>();
    const [timePeriod, setTimePeriod] = useState<'1w' | '1m' | '3m' | '6m' | '12m'>('1m');
    const { ooo: oooRecords } = useSelector((state: RootState) => state.records);
    const { ooo: oooForecasts } = useSelector((state: RootState) => state.forecast);

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
        <div className="component-ui-ooo-room-graph-container">
            <div className="component-ui-ooo-room-graph-header">
                <h3 className="component-ui-ooo-room-graph-title">OOO Room Analysis</h3>
                <div className="component-ui-ooo-room-graph-toggle">
                    {(['1w', '1m', '3m', '6m', '12m'] as const).map(period => (
                        <button
                            key={period}
                            className={`component-ui-ooo-room-graph-toggle-btn ${timePeriod === period ? 'active' : ''}`}
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
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="date" tick={{ fill: '#666', fontSize: 12 }} interval={xAxisInterval} />
                        <YAxis tick={{ fill: '#666', fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#fff',
                                border: '1px solid #e0e0e0',
                                borderRadius: '4px',
                                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }}
                            formatter={(value, name) => {
                                if (name === 'actual') return [value, 'Actual OOO Rooms'];
                                return [value, 'Predicted OOO Rooms'];
                            }}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        <Bar dataKey="actual" fill="#e74c3c" name="Actual OOO Rooms" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="predicted" fill="#f39c12" name="Predicted OOO Rooms" radius={[4, 4, 0, 0]} />
                    </BarChart>
                ) : (
                    <LineChart data={oooData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis dataKey="date" tick={{ fill: '#666', fontSize: 12 }} interval={xAxisInterval} />
                        <YAxis tick={{ fill: '#666', fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#fff',
                                border: '1px solid #e0e0e0',
                                borderRadius: '4px',
                                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                            }}
                            formatter={(value, name) => {
                                if (name === 'actual') return [value, 'Actual OOO Rooms'];
                                return [value, 'Predicted OOO Rooms'];
                            }}
                        />
                        <Legend wrapperStyle={{ paddingTop: '20px' }} />
                        <Line
                            type="monotone"
                            dataKey="actual"
                            stroke="#e74c3c"
                            dot={false}
                            strokeWidth={2}
                            name="Actual OOO Rooms"
                        />
                        <Line
                            type="monotone"
                            dataKey="predicted"
                            stroke="#f39c12"
                            dot={false}
                            strokeWidth={2}
                            strokeDasharray="5 5"
                            name="Predicted OOO Rooms"
                        />
                    </LineChart>
                )}
            </ResponsiveContainer>
        </div>
    );
};

export default OOORoomGraph;
