"use client"
import { useEffect, useMemo, useState } from "react"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowUpRight, ArrowDownRight, TrendingUp } from "lucide-react"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"

type Metric = { label: string; value: string; change?: string; trend?: "up" | "down" }

type Activity = { vendor: string; amount: string; category: string; dateText: string }

type CashPoint = { date: string; value: number }

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"

export function AnalysisOverview() {
  const [trend, setTrend] = useState<CashPoint[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>("")
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [activity, setActivity] = useState<Activity[]>([])

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true)
        setError("")
        const res = await fetch(`${API_BASE}/get-current-data`, { cache: "no-store" })
        if (!res.ok) throw new Error(`API ${res.status}`)
        const data = await res.json()
        const rows: any[] = Array.isArray(data?.bank_data) ? data.bank_data : []

        // Aggregate by day: net cash flow per date
        const map = new Map<string, number>()
        for (const r of rows) {
          const rawDate = r.Date || r["Transaction Date"] || r.date || ""
          const d = String(rawDate).slice(0, 10) // YYYY-MM-DD
          const amt = Number(r.Amount ?? r["Credit Amount"] ?? r["Debit Amount"] ?? r.amount ?? 0)
          map.set(d, (map.get(d) || 0) + amt)
        }

        // Sort by date asc and keep last 30
        const points: CashPoint[] = Array.from(map.entries())
          .sort((a, b) => (a[0] < b[0] ? -1 : 1))
          .map(([date, value]) => ({ date, value }))
          .slice(-30)

        setTrend(points)

        // Build overview metrics dynamically
        const categorize = (desc: string, cat: string) => {
          const lc = String(cat || '').toLowerCase()
          const descLc = String(desc || '').toLowerCase()
          if (lc.includes('operat')) return 'Operating Activities'
          if (lc.includes('invest')) return 'Investing Activities'
          if (lc.includes('financ') || /(loan|interest|equity|dividend|debt|bank)/.test(descLc)) return 'Financing Activities'
          if (/(salary|logistics|shipping|maintenance|utility|power|fuel|freight|coal)/.test(descLc)) return 'Operating Activities'
          if (/(equipment|plant|machinery|construction|contractor|project|capex)/.test(descLc)) return 'Investing Activities'
          return 'Uncategorized'
        }

        let op = 0, inv = 0, fin = 0
        for (const r of rows) {
          const desc = r.Description || r["Transaction Description"] || r.description || ""
          const cat = r.Category || r.category || ''
          const bucket = categorize(desc, cat)
          if (bucket === 'Operating Activities') op++
          else if (bucket === 'Investing Activities') inv++
          else if (bucket === 'Financing Activities') fin++
        }

        setMetrics([
          { label: 'Total Transactions', value: (rows.length || 0).toLocaleString() },
          { label: 'Operating Activities', value: op.toLocaleString() },
          { label: 'Investing Activities', value: inv.toLocaleString() },
          { label: 'Financing Activities', value: fin.toLocaleString() },
        ])

        // Recent Activity: pick 3 most recent by date
        const parseDate = (val: any) => new Date(String(val || '').replace(/\//g, '-'))
        const tx = rows.map(r => ({
          date: parseDate(r.Date || r["Transaction Date"] || r.date),
          description: String(r.Description || r["Transaction Description"] || r.description || ''),
          vendor: String(r.Vendor || r.vendor || '—'),
          amount: Number(r.Amount ?? r["Credit Amount"] ?? r["Debit Amount"] ?? r.amount ?? 0),
          category: categorize(r.Description || r["Transaction Description"] || r.description || '', r.Category || r.category || ''),
        }))
        .filter(t => !isNaN(t.date.getTime()))
        .sort((a,b) => b.date.getTime() - a.date.getTime())
        .slice(0,3)

        const relTime = (d: Date) => {
          const diffMs = Date.now() - d.getTime()
          const mins = Math.round(diffMs / 60000)
          if (mins < 60) return `${mins} min ago`
          const hrs = Math.round(mins / 60)
          if (hrs < 24) return `${hrs} hour${hrs>1?'s':''} ago`
          const days = Math.round(hrs / 24)
          return `${days} day${days>1?'s':''} ago`
        }

        setActivity(tx.map(t => ({
          vendor: t.vendor && t.vendor !== '-' ? t.vendor : (t.description || 'Unknown'),
          amount: `${t.amount >= 0 ? '' : '-'}$${Math.abs(t.amount).toLocaleString()}`,
          category: t.category.includes('Operating') ? 'Operating' : t.category.includes('Investing') ? 'Investing' : t.category.includes('Financing') ? 'Financing' : 'Other',
          dateText: relTime(t.date),
        })))
      } catch (e: any) {
        setError(e?.message || "Failed to load chart data")
        setTrend([])
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const yDomain = useMemo(() => {
    if (trend.length === 0) return undefined
    const vals = trend.map(p => p.value)
    const min = Math.min(...vals)
    const max = Math.max(...vals)
    const pad = Math.max(1, Math.round((max - min) * 0.1))
    return [Math.floor(min - pad), Math.ceil(max + pad)] as [number, number]
  }, [trend])

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2 space-y-6">
        <div>
          <h2 className="text-xl font-semibold mb-4">Overview</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {metrics.map((metric) => (
              <Card key={metric.label} className="p-6">
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">{metric.label}</span>
                    {metric.change && metric.trend && (
                      <Badge variant={metric.trend === "up" ? "default" : "secondary"} className="gap-1">
                        {metric.trend === "up" ? (
                          <ArrowUpRight className="h-3 w-3" />
                        ) : (
                          <ArrowDownRight className="h-3 w-3" />
                        )}
                        {metric.change}
                      </Badge>
                    )}
                  </div>
                  <div className="text-2xl font-semibold">{metric.value}</div>
                </div>
              </Card>
            ))}
          </div>
        </div>

        <Card className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <TrendingUp className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold">Cash Flow Trend</h3>
              <p className="text-sm text-muted-foreground">Last 30 days</p>
            </div>
          </div>
          <div className="h-64">
            {loading && (
              <div className="h-full flex items-center justify-center bg-muted/30 rounded-lg text-sm text-muted-foreground">
                Loading chart…
              </div>
            )}
            {!loading && trend.length === 0 && (
              <div className="h-full flex items-center justify-center bg-muted/30 rounded-lg text-sm text-muted-foreground">
                {error ? error : "No data available. Try uploading transactions."}
              </div>
            )}
            {!loading && trend.length > 0 && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trend} margin={{ left: 8, right: 8, top: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 12 }} minTickGap={16} />
                  <YAxis domain={yDomain} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={(v: any) => `$${Number(v).toLocaleString()}`} />
                  <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
        <Card className="p-6">
          <div className="space-y-4">
            {activity.map((a, index) => (
              <div key={index} className="flex flex-col gap-2 pb-4 border-b border-border last:border-0 last:pb-0">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{a.vendor}</p>
                    <p className="text-sm text-muted-foreground">{a.category}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold">{a.amount}</p>
                    <p className="text-xs text-muted-foreground">{a.dateText}</p>
                  </div>
                </div>
              </div>
            ))}
            {activity.length === 0 && (
              <div className="text-sm text-muted-foreground">No recent transactions found.</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
