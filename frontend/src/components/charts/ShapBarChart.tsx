import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from 'recharts'

import { formatFeatureName } from '@/lib/format'
import { TARGET_HIGHER_IS_BETTER, type ShapContribution, type Target } from '@/lib/types'

export function ShapBarChart({
  contributions,
  target,
}: {
  contributions: ShapContribution[]
  target: Target
}) {
  const higherIsBetter = TARGET_HIGHER_IS_BETTER[target]
  const data = [...contributions]
    .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    .reverse()
    .map((c) => ({
      name: formatFeatureName(c.feature),
      shap: c.shap,
      good: higherIsBetter ? c.shap > 0 : c.shap < 0,
    }))

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.shap)), 0.01)

  return (
    <div style={{ height: data.length * 36 + 24 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
          <XAxis type="number" domain={[-maxAbs, maxAbs]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={168}
            tickLine={false}
            axisLine={false}
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
          />
          <ReferenceLine x={0} stroke="var(--border-default)" />
          <Bar dataKey="shap" radius={2} barSize={16}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.good ? 'var(--chart-positive)' : 'var(--chart-negative)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
