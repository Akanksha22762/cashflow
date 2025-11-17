"use client";

import { useEffect, useMemo, useState } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";

type Row = Record<string, any>;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

function toCsv(rows: Row[]): string {
  if (rows.length === 0) return "";
  const headersSet: Set<string> = rows.reduce<Set<string>>((set, r) => {
    Object.keys(r).forEach((k) => set.add(k));
    return set;
  }, new Set<string>());
  const headers = Array.from(headersSet);
  const esc = (v: any) => {
    const s = String(v ?? "");
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  const lines = [headers.join(","), ...rows.map((r) => headers.map((h) => esc(r[h])).join(","))];
  return lines.join("\n");
}

function toJson(obj: any): string {
  return JSON.stringify(obj, null, 2);
}

export default function ReportsPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError("");
        const res = await fetch(`${API_BASE}/get-current-data`, { cache: "no-store" });
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();
        const list: any[] = Array.isArray(data?.bank_data) ? data.bank_data : [];
        setRows(list);
      } catch (e: any) {
        setError(e?.message || "Failed to load data");
        setRows([]);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const summary = useMemo(() => {
    const amountOf = (r: Row) => Number(r.Amount ?? r["Credit Amount"] ?? r["Debit Amount"] ?? r.amount ?? 0);
    const vendorOf = (r: Row) => String(r.Vendor || r.vendor || "Unknown");
    const totals = rows.reduce(
      (acc: { inflows: number; outflows: number }, r) => {
        const a = amountOf(r);
        if (a >= 0) acc.inflows += a;
        else acc.outflows += Math.abs(a);
        return acc;
      },
      { inflows: 0, outflows: 0 }
    );
    const net: number = totals.inflows - totals.outflows;
    // Build vendor aggregates for export
    const vendorAgg = new Map<string, { total: number; transactions: number }>();
    rows.forEach((r) => {
      const v = vendorOf(r);
      const a = amountOf(r);
      const current = vendorAgg.get(v) || { total: 0, transactions: 0 };
      current.total += a;
      current.transactions += 1;
      vendorAgg.set(v, current);
    });
    // Build analytics time-series for export (daily net)
    const dateOf = (r: Row) => String(r.Date || r["Transaction Date"] || r.date || "").slice(0, 10);
    const byDate = new Map<string, number>();
    rows.forEach((r) => {
      const d = dateOf(r);
      if (!d) return;
      const a = amountOf(r);
      byDate.set(d, (byDate.get(d) || 0) + a);
    });
    const series = Array.from(byDate.entries())
      .sort((a, b) => (a[0] < b[0] ? -1 : 1))
      .map(([date, net]) => ({ date, net }));
    return { totals: { inflows: totals.inflows, outflows: totals.outflows, net }, count: rows.length, vendorAgg, series };
  }, [rows]);

  const download = (filename: string, content: string, contentType: string = "text/csv;charset=utf-8;") => {
    const blob = new Blob([content], { type: contentType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const downloadTransactions = () => download("transactions_export.csv", toCsv(rows));

  const downloadVendors = () => {
    const vendorRows: Row[] = Array.from(summary.vendorAgg.entries()).map(([vendor, v]) => ({ vendor, total: v.total, transactions: v.transactions }));
    download("vendors_export.csv", toCsv(vendorRows));
  };

  const downloadAnalytics = () => {
    const seriesRows: Row[] = summary.series.map((p) => ({ date: p.date, net: p.net }));
    download("analytics_timeseries.csv", toCsv(seriesRows));
  };

  const downloadComprehensive = async () => {
    try {
      setLoading(true);
      setError("");
      const res = await fetch(`${API_BASE}/comprehensive-report`, { cache: "no-store" });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const comprehensiveData = await res.json();
      
      if (!comprehensiveData.success) {
        throw new Error(comprehensiveData.error || "Failed to generate comprehensive report");
      }

      // Create multi-sheet CSV export or detailed JSON
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      
      // Download as JSON for complete data preservation
      download(
        `comprehensive_report_${timestamp}.json`,
        toJson(comprehensiveData),
        "application/json;charset=utf-8;"
      );

      // Also create separate CSV files for each section
      if (comprehensiveData.transactions && comprehensiveData.transactions.length > 0) {
        download(`comprehensive_transactions_${timestamp}.csv`, toCsv(comprehensiveData.transactions));
      }
      
      if (comprehensiveData.vendor_analysis && comprehensiveData.vendor_analysis.length > 0) {
        download(`comprehensive_vendors_${timestamp}.csv`, toCsv(comprehensiveData.vendor_analysis));
      }
      
      if (comprehensiveData.category_analysis && comprehensiveData.category_analysis.length > 0) {
        download(`comprehensive_categories_${timestamp}.csv`, toCsv(comprehensiveData.category_analysis));
      }
      
      if (comprehensiveData.analytics_timeseries && comprehensiveData.analytics_timeseries.length > 0) {
        download(`comprehensive_analytics_${timestamp}.csv`, toCsv(comprehensiveData.analytics_timeseries));
      }
      
      if (comprehensiveData.monthly_patterns && comprehensiveData.monthly_patterns.length > 0) {
        download(`comprehensive_monthly_patterns_${timestamp}.csv`, toCsv(comprehensiveData.monthly_patterns));
      }

      // Executive summary as separate CSV
      if (comprehensiveData.executive_summary) {
        const summaryRows = [comprehensiveData.executive_summary];
        download(`comprehensive_executive_summary_${timestamp}.csv`, toCsv(summaryRows));
      }

      alert("✅ Comprehensive report downloaded successfully!\n\nDownloaded files:\n- Complete report (JSON)\n- Transactions (CSV)\n- Vendors (CSV)\n- Categories (CSV)\n- Analytics (CSV)\n- Monthly Patterns (CSV)\n- Executive Summary (CSV)");
      
    } catch (e: any) {
      setError(e?.message || "Failed to download comprehensive report");
    } finally {
      setLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-end">
          <button 
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={async () => {
              try {
                setLoading(true);
                setError("");
                const res = await fetch(`${API_BASE}/comprehensive-report.pdf`, { cache: "no-store" });
                if (!res.ok) throw new Error(`API ${res.status}`);
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `comprehensive_report_${new Date().toISOString().replace(/[:.]/g,'-')}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
              } catch (e: any) {
                setError(e?.message || 'Failed to download PDF');
              } finally {
                setLoading(false);
              }
            }}
            disabled={rows.length === 0 || loading}
          >
            {loading ? 'Generating PDF...' : 'Download Report (PDF)'}
          </button>
        </div>

        {loading && <div>Loading…</div>}
        {error && <div className="text-sm text-red-600">{error}</div>}

        {!loading && !error && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="p-6 border rounded-lg bg-card">
              <h2 className="text-lg font-semibold mb-4">Executive Summary</h2>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border rounded-md bg-background">
                  <div className="text-sm text-muted-foreground">Total Transactions</div>
                  <div className="text-xl font-semibold">{summary.count.toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-background">
                  <div className="text-sm text-muted-foreground">Inflows</div>
                  <div className="text-xl font-semibold text-green-600">${summary.totals.inflows.toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-background">
                  <div className="text-sm text-muted-foreground">Outflows</div>
                  <div className="text-xl font-semibold text-red-600">${summary.totals.outflows.toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-background">
                  <div className="text-sm text-muted-foreground">Net Cash Flow</div>
                  <div className={`text-xl font-semibold ${summary.totals.net >= 0 ? 'text-green-600' : 'text-red-600'}`}>${Math.abs(summary.totals.net).toLocaleString()}</div>
                </div>
              </div>
            </div>

            <div className="p-6 border rounded-lg bg-card">
              <h2 className="text-lg font-semibold mb-4">Preview (first 10 rows)</h2>
              <div className="overflow-auto border rounded-md max-h-[420px]">
                <table className="min-w-full text-sm">
                  <thead className="bg-muted sticky top-0">
                    <tr>
                      {Array.from(rows[0] ? Object.keys(rows[0]) : []).map((h) => (
                        <th key={h} className="px-3 py-2 text-left whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(rows.slice(0, 10)).map((r, idx) => (
                      <tr key={idx} className="border-t">
                        {Array.from(rows[0] ? Object.keys(rows[0]) : []).map((h) => (
                          <td key={h} className="px-3 py-2 whitespace-nowrap">{String(r[h] ?? "")}</td>
                        ))}
                      </tr>
                    ))}
                    {rows.length === 0 && (
                      <tr>
                        <td className="px-3 py-6 text-center text-muted-foreground" colSpan={5}>No data to preview.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}


