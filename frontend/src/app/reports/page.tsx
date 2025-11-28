"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";

type Row = Record<string, any>;

type VendorAnalysis = {
  vendor_name: string;
  inflows: number;
  outflows: number;
  net_cash_flow: number;
  transaction_count: number;
  // ✅ Removed opening_balance and closing_balance - these are account-level, not vendor-level
  metadata?: Record<string, any>;
};

type ComprehensiveReport = {
  success: boolean;
  generated_at?: string;
  error?: string; // ✅ Added error field for error responses
  summary?: {
    opening_balance?: number;
    total_inflows: number;
    total_outflows: number;
    net_cash_flow: number;
    closing_balance?: number;
    final_closing_balance?: number;
    transaction_count: number;
  };
  transactions?: Row[];
  vendor_analysis?: VendorAnalysis[];
  category_analysis?: Array<{
    category: string;
    inflows: number;
    outflows: number;
    net: number;
    transaction_count: number;
  }>;
  analytics_timeseries?: Array<{ date: string; net: number }>;
  monthly_patterns?: Array<Record<string, any>>;
  executive_summary?: Record<string, any>;
  cashflow_statement?: {
    period?: string;
    period_start?: string;
    period_end?: string;
    operating_activities?: {
      inflow_items?: Record<string, number>;
      total_inflows: number;
      outflow_items?: Record<string, number>;
      total_outflows: number;
      net_cash_flow: number;
    };
    investing_activities?: {
      inflow_items?: Record<string, number>;
      total_inflows: number;
      outflow_items?: Record<string, number>;
      total_outflows: number;
      net_cash_flow: number;
    };
    financing_activities?: {
      inflow_items?: Record<string, number>;
      total_inflows: number;
      outflow_items?: Record<string, number>;
      total_outflows: number;
      net_cash_flow: number;
    };
    net_increase_in_cash?: number;
    opening_cash_balance?: number;
    closing_cash_balance?: number;
  };
};

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
  const [report, setReport] = useState<ComprehensiveReport | null>(null);
  const [transactions, setTransactions] = useState<Row[]>([]);
  const [vendors, setVendors] = useState<VendorAnalysis[]>([]);
  const [analyticsSeries, setAnalyticsSeries] = useState<Array<{ date: string; net: number }>>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [hasLoaded, setHasLoaded] = useState<boolean>(false);
  const isLoadingRef = useRef<boolean>(false); // Prevent duplicate requests
  const abortControllerRef = useRef<AbortController | null>(null);

  const loadReports = async () => {
    // Prevent duplicate simultaneous requests
    if (isLoadingRef.current) {
      console.log(`[REPORTS] Request already in progress, skipping duplicate...`);
      return;
    }

    // Cancel any previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    isLoadingRef.current = true;
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    
    try {
      setLoading(true);
      setError("");
      console.log(`[REPORTS] Fetching from: ${API_BASE}/comprehensive-report`);
      const res = await fetch(`${API_BASE}/comprehensive-report`, { 
        cache: "no-store",
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        signal: abortController.signal
      });
      
      console.log(`[REPORTS] Response status: ${res.status} ${res.statusText}`);
      
      if (!res.ok) {
        const errorText = await res.text().catch(() => 'Unable to read error response');
        console.error(`[REPORTS] API error response:`, errorText);
        throw new Error(`API ${res.status}: ${errorText.substring(0, 100)}`);
      }
      
      const contentType = res.headers.get('content-type');
      console.log(`[REPORTS] Content-Type: ${contentType}`);
      
      if (!contentType || !contentType.includes('application/json')) {
        const text = await res.text().catch(() => 'Unable to read response');
        console.error(`[REPORTS] Non-JSON response:`, text.substring(0, 200));
        throw new Error(`Expected JSON but got ${contentType}. Response: ${text.substring(0, 100)}`);
      }
      
      const data: ComprehensiveReport = await res.json();
      console.log(`[REPORTS] Received data, success: ${data?.success}`);
      
      // Debug: Log cash flow statement
      console.log(`[REPORTS] Cash flow statement present:`, !!data?.cashflow_statement);
      if (data?.cashflow_statement) {
        console.log(`[REPORTS] Cash flow statement keys:`, Object.keys(data.cashflow_statement));
      }
      
      if (!data?.success) {
        const errorMsg = data?.executive_summary?.error || data?.error || "Failed to build report";
        console.error(`[REPORTS] Report generation failed:`, errorMsg);
        throw new Error(errorMsg);
      }
      
      console.log(`[REPORTS] Report loaded successfully`);
      setReport(data);
      setTransactions(Array.isArray(data?.transactions) ? data.transactions : []);
      setVendors(Array.isArray(data?.vendor_analysis) ? data.vendor_analysis : []);
      setAnalyticsSeries(Array.isArray(data?.analytics_timeseries) ? data.analytics_timeseries : []);
      setHasLoaded(true); // Mark as loaded (for display purposes)
      setError(""); // Clear any previous errors
    } catch (e: any) {
      // Don't show error if request was cancelled
      if (e?.name === 'AbortError') {
        console.log(`[REPORTS] Request cancelled`);
        return;
      }
      // Don't show error if connection reset after successful load
      if (e?.message?.includes('Failed to fetch') && hasLoaded) {
        console.log(`[REPORTS] Connection error after successful load, ignoring`);
        return;
      }
      console.error(`[REPORTS] Error loading report:`, e);
      setError(e?.message || "Failed to load data");
    } finally {
      setLoading(false);
      isLoadingRef.current = false;
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null;
      }
    }
  };

  useEffect(() => {
    // Load when component mounts
    loadReports();
    
    // Cleanup on unmount - cancel any pending requests
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      isLoadingRef.current = false;
    };
  }, []); // Only run once on mount

  // Function to manually refresh reports (useful after data updates like category changes)
  const refreshReports = async () => {
    setHasLoaded(false);
    await loadReports();
  };

  const summary = useMemo(() => {
    if (!report?.summary) {
      return { totals: { inflows: 0, outflows: 0, net: 0, opening: 0, closing: 0 }, count: 0 };
    }
    const inflows = report.summary.total_inflows ?? 0;
    const outflows = report.summary.total_outflows ?? 0;
    const net = report.summary.net_cash_flow ?? 0;
    const closing = report.summary.closing_balance ?? report.summary.final_closing_balance ?? 0;
    const opening = report.summary.opening_balance ?? (closing ? closing - net : 0);
    const count = report.summary.transaction_count ?? transactions.length;
    return { totals: { inflows, outflows, net, opening, closing }, count };
  }, [report, transactions.length]);

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

  const downloadTransactions = () => download("transactions_export.csv", toCsv(transactions));

  const downloadVendors = () => download("vendors_export.csv", toCsv(vendors));

  const downloadAnalytics = () => download("analytics_timeseries.csv", toCsv(analyticsSeries));

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
        <div className="flex flex-col gap-3 items-stretch justify-end md:flex-row md:items-center md:justify-end">
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
            disabled={transactions.length === 0 || loading}
          >
            {loading ? 'Generating PDF...' : 'Download Report (PDF)'}
          </button>
          <div className="flex flex-wrap gap-2">
            <button
              className="px-4 py-2 bg-green-600 text-white rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={refreshReports}
              disabled={loading}
              title="Refresh report with latest data (useful after updating categories)"
            >
              {loading ? "Refreshing..." : "🔄 Refresh Report"}
            </button>
            <button
              className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={downloadTransactions}
              disabled={transactions.length === 0 || loading}
            >
              Export Transactions (CSV)
            </button>
            <button
              className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={downloadVendors}
              disabled={vendors.length === 0 || loading}
            >
              Export Vendors (CSV)
            </button>
            <button
              className="px-4 py-2 bg-secondary text-secondary-foreground rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={downloadAnalytics}
              disabled={analyticsSeries.length === 0 || loading}
            >
              Export Analytics (CSV)
            </button>
          </div>
        </div>

        {loading && <div>Loading…</div>}
        {error && <div className="text-sm text-red-600">{error}</div>}

        {!loading && !error && report && (
          <div className="p-6 border rounded-lg bg-card">
            <h2 className="text-lg font-semibold mb-4">Executive Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 border rounded-md bg-background">
                <div className="text-sm text-muted-foreground">Total Transactions</div>
                <div className="text-xl font-semibold">{summary.count.toLocaleString()}</div>
              </div>
              <div className="p-3 border rounded-md bg-background">
                <div className="text-sm text-muted-foreground">Inflows</div>
                <div className="text-xl font-semibold text-green-600">${summary.totals.inflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
              <div className="p-3 border rounded-md bg-background">
                <div className="text-sm text-muted-foreground">Outflows</div>
                <div className="text-xl font-semibold text-red-600">${summary.totals.outflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
              <div className="p-3 border rounded-md bg-background">
                <div className="text-sm text-muted-foreground">Net Cash Flow</div>
                <div className={`text-xl font-semibold ${summary.totals.net >= 0 ? 'text-green-600' : 'text-red-600'}`}>${Math.abs(summary.totals.net).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && report && report.cashflow_statement && (
          <div className="p-6 border rounded-lg bg-card">
            <h2 className="text-xl font-bold mb-2">CASH FLOW STATEMENT – DIRECT METHOD</h2>
            {report.cashflow_statement.period && (
              <div className="text-sm text-muted-foreground mb-6">For the Period: {report.cashflow_statement.period}</div>
            )}
            
            <div className="space-y-6 font-mono text-sm">
              {/* Operating Activities */}
              <div>
                <div className="font-bold mb-2">A. CASH FLOWS FROM OPERATING ACTIVITIES</div>
                <div className="ml-4 space-y-1">
                  {report.cashflow_statement.operating_activities?.inflow_items && 
                   Object.entries(report.cashflow_statement.operating_activities.inflow_items).map(([item, amount]) => (
                    <div key={item} className="flex justify-between">
                      <span className="ml-4">{item}</span>
                      <span className="text-right">{amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                  {report.cashflow_statement.operating_activities?.inflow_items && 
                   Object.keys(report.cashflow_statement.operating_activities.inflow_items).length > 0 && (
                    <div className="border-t my-1"></div>
                  )}
                  <div className="flex justify-between font-semibold">
                    <span>Total Cash Inflows</span>
                    <span className="text-right">{report.cashflow_statement.operating_activities?.total_inflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                  </div>
                  <div className="mb-2"></div>
                  {report.cashflow_statement.operating_activities?.outflow_items && 
                   Object.entries(report.cashflow_statement.operating_activities.outflow_items).map(([item, amount]) => (
                    <div key={item} className="flex justify-between">
                      <span className="ml-4">{item}</span>
                      <span className="text-right">{amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                  {report.cashflow_statement.operating_activities?.outflow_items && 
                   Object.keys(report.cashflow_statement.operating_activities.outflow_items).length > 0 && (
                    <div className="border-t my-1"></div>
                  )}
                  <div className="flex justify-between font-semibold">
                    <span>Total Cash Outflows</span>
                    <span className="text-right">{report.cashflow_statement.operating_activities?.total_outflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                  </div>
                  <div className="border-t mt-2 mb-2"></div>
                  <div className="flex justify-between font-semibold">
                    <span>Net Cash from Operating Activities</span>
                    <span className="text-right">{report.cashflow_statement.operating_activities?.net_cash_flow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                  </div>
                </div>
                <div className="border-t my-4"></div>
              </div>

              {/* Investing Activities */}
              <div>
                <div className="font-bold mb-2">B. CASH FLOWS FROM INVESTING ACTIVITIES</div>
                <div className="ml-4 space-y-1">
                  {report.cashflow_statement.investing_activities?.outflow_items && 
                   Object.entries(report.cashflow_statement.investing_activities.outflow_items).map(([item, amount]) => (
                    <div key={item} className="flex justify-between">
                      <span className="ml-4">{item}</span>
                      <span className="text-right">{amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                  <div className="border-t mt-2 mb-2"></div>
                  <div className="flex justify-between font-semibold">
                    <span>Net Cash from Investing Activities</span>
                    <span className="text-right">{report.cashflow_statement.investing_activities?.net_cash_flow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                  </div>
                </div>
                <div className="border-t my-4"></div>
              </div>

              {/* Financing Activities */}
              <div>
                <div className="font-bold mb-2">C. CASH FLOWS FROM FINANCING ACTIVITIES</div>
                <div className="ml-4 space-y-1">
                  {report.cashflow_statement.financing_activities?.inflow_items && 
                   Object.entries(report.cashflow_statement.financing_activities.inflow_items).map(([item, amount]) => (
                    <div key={item} className="flex justify-between">
                      <span className="ml-4">{item}</span>
                      <span className="text-right">{amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                  {report.cashflow_statement.financing_activities?.outflow_items && 
                   Object.entries(report.cashflow_statement.financing_activities.outflow_items).map(([item, amount]) => (
                    <div key={item} className="flex justify-between">
                      <span className="ml-4">{item}</span>
                      <span className="text-right">{amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</span>
                    </div>
                  ))}
                  <div className="border-t mt-2 mb-2"></div>
                  <div className="flex justify-between font-semibold">
                    <span>Net Cash from Financing Activities</span>
                    <span className="text-right">{report.cashflow_statement.financing_activities?.net_cash_flow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                  </div>
                </div>
                <div className="border-t my-4"></div>
              </div>

              {/* Summary */}
              <div className="space-y-2">
                <div className="flex justify-between font-semibold">
                  <span>Net Increase in Cash</span>
                  <span className="text-right">{report.cashflow_statement.net_increase_in_cash?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}</span>
                </div>
                <div className="flex justify-between">
                  <span>Opening Cash Balance</span>
                  <span className="text-right">{report.cashflow_statement.opening_cash_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}</span>
                </div>
                <div className="flex justify-between font-semibold">
                  <span>Closing Cash Balance</span>
                  <span className="text-right">{report.cashflow_statement.closing_cash_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}</span>
                </div>
                <div className="border-t my-4"></div>
              </div>
            </div>
          </div>
        )}

        {!loading && !error && report && vendors.length > 0 && (
          <div className="p-6 border rounded-lg bg-card">
            <h3 className="text-lg font-semibold mb-4">Vendor Breakdown</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-muted-foreground border-b">
                    <th className="py-2 pr-4">Vendor</th>
                    <th className="py-2 pr-4 text-right">Inflows</th>
                    <th className="py-2 pr-4 text-right">Outflows</th>
                    <th className="py-2 pr-4 text-right">Net</th>
                    <th className="py-2 pr-4 text-right">Transactions</th>
                  </tr>
                </thead>
                <tbody>
                  {vendors.slice(0, 8).map((vendor) => (
                    <tr key={vendor.vendor_name} className="border-b last:border-0">
                      <td className="py-2 pr-4 font-medium">{vendor.vendor_name}</td>
                      <td className="py-2 pr-4 text-right text-green-600">₹{vendor.inflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="py-2 pr-4 text-right text-red-600">₹{vendor.outflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className={`py-2 pr-4 text-right ${vendor.net_cash_flow >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ₹{vendor.net_cash_flow.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td className="py-2 pr-4 text-right">{vendor.transaction_count.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!loading && !error && report && report.category_analysis && report.category_analysis.length > 0 && (
          <div className="p-6 border rounded-lg bg-card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Category Analysis</h3>
              <button className="text-sm text-blue-600 hover:underline" onClick={() => download("category_analysis.csv", toCsv(report.category_analysis || []))}>
                Export CSV
              </button>
            </div>
            <div className="grid gap-3">
              {report.category_analysis.map((category) => (
                <div key={category.category} className="p-4 border rounded-md bg-background">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm text-muted-foreground">Category</div>
                      <div className="text-base font-semibold">{category.category}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-muted-foreground">Net</div>
                      <div className={`text-lg font-semibold ${category.net >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ₹{category.net.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mt-3 text-sm">
                    <div>
                      <div className="text-muted-foreground">Inflows</div>
                      <div className="text-green-600 font-medium">₹{category.inflows.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Outflows</div>
                      <div className="text-red-600 font-medium">₹{category.outflows.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-muted-foreground">Transactions</div>
                      <div className="font-medium">{category.transaction_count.toLocaleString()}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}


