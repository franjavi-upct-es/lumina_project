// frontend/src/components/dashboard/EquityCurve.tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export function EquityCurve({ data }: { data: { time: string; equity: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="equity" stroke="#8884d8" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
