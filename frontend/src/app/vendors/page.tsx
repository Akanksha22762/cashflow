"use client";

import { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle, Building2, DollarSign, Calendar, Activity } from "lucide-react";

type Vendor = {
  name: string;
  count?: number;
  inflow?: number;
  outflow?: number;
};

type VendorAnalysis = {
  status: string;
  message: string;
  // Backend returns a root-level reasoning_explanations object
  reasoning_explanations?: any;
  data: {
    // Map shape when multiple vendors are returned
    vendor_analysis?: any;
    // Single-vendor shape fields
    vendor_name?: string;
    analysis_summary?: {
      total_transactions?: number;
      net_cash_flow?: number;
      avg_transaction?: number;
    };
    total_amount?: number;
    transaction_count?: number;
    payment_frequency?: string;
    vendor_importance?: string;
    cash_flow_categories?: Record<string, number>;
    recommendations?: string[];
    historical_data?: Array<{ period: string; amount: number }>;
  };
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

// Component to display vendor analysis results
function VendorAnalysisDisplay({ analysis, vendorName }: { analysis: VendorAnalysis; vendorName: string }) {
  // Show backend error explicitly
  const backendError = (analysis as any)?.error || (analysis?.status === 'error' ? analysis?.message || 'Analysis failed' : '');
  if (backendError) {
    return (
      <div className="p-4 border rounded-md bg-red-50 text-red-700 text-sm">
        {String(backendError)}
      </div>
    );
  }

  // Primary shape: data.vendor_analysis[vendorName] (case-insensitive key match)
  let vendorData: any = undefined;
  const map: any = analysis.data?.vendor_analysis;
  if (map && typeof map === 'object') {
    const target = vendorName?.toLowerCase().trim();
    const matchedKey = Object.keys(map).find((k) => k.toLowerCase().trim() === target);
    vendorData = matchedKey ? map[matchedKey] : undefined;
  }

  // Also support single-vendor shape: data.{vendor_name, analysis_summary, ...}
  if (!vendorData && analysis?.data?.vendor_name && String(analysis.data.vendor_name).toLowerCase().trim() === vendorName.toLowerCase().trim()) {
    const a = analysis.data as any;
    vendorData = {
      analysis: {
        total_transactions: a.analysis_summary?.total_transactions ?? a.transaction_count ?? 0,
        net_amount: a.analysis_summary?.net_cash_flow ?? a.total_amount ?? 0,
        payment_frequency: a.payment_frequency ?? 'N/A',
        vendor_importance: a.vendor_importance ?? 'N/A'
      },
      cash_flow_categories: a.cash_flow_categories,
      insights: analysis.reasoning_explanations?.simple_reasoning ? [analysis.reasoning_explanations.simple_reasoning] : [],
      recommendations: a.recommendations,
      historical_data: a.historical_data,
      ai_analysis: analysis.reasoning_explanations?.ai_analysis,
      ml_analysis: analysis.reasoning_explanations?.ml_analysis
    };
  }

  // Permissive fallback: if backend returned a single-vendor shaped payload without vendor_name, still render
  if (!vendorData && analysis?.data && (analysis.data as any).analysis_summary) {
    const a = analysis.data as any;
    vendorData = {
      analysis: {
        total_transactions: a.analysis_summary?.total_transactions ?? a.transaction_count ?? 0,
        net_amount: a.analysis_summary?.net_cash_flow ?? a.total_amount ?? 0,
        payment_frequency: a.payment_frequency ?? 'N/A',
        vendor_importance: a.vendor_importance ?? 'N/A'
      },
      cash_flow_categories: a.cash_flow_categories,
      insights: analysis.reasoning_explanations?.simple_reasoning ? [analysis.reasoning_explanations.simple_reasoning] : [],
      recommendations: a.recommendations,
      historical_data: a.historical_data,
      ai_analysis: analysis.reasoning_explanations?.ai_analysis,
      ml_analysis: analysis.reasoning_explanations?.ml_analysis
    };
  }

  // Legacy payload support: some endpoints return data.summary_cards structure
  if (!vendorData && analysis?.data && (analysis.data as any).summary_cards) {
    const a = (analysis.data as any);
    const tx = a.summary_cards?.transactions?.value ?? a.transaction_count ?? 0;
    const net = a.summary_cards?.net_cash_flow?.value ?? a.total_amount ?? 0;
    vendorData = {
      analysis: {
        total_transactions: tx,
        net_amount: net,
        payment_frequency: a.payment_frequency ?? 'N/A',
        vendor_importance: a.vendor_importance ?? 'N/A'
      },
      cash_flow_categories: a.cash_flow_categories,
      insights: analysis.reasoning_explanations?.simple_reasoning ? [analysis.reasoning_explanations.simple_reasoning] : [],
      recommendations: a.recommendations,
      historical_data: a.historical_data,
      ai_analysis: analysis.reasoning_explanations?.ai_analysis,
      ml_analysis: analysis.reasoning_explanations?.ml_analysis
    };
  }

  if (!vendorData) {
    return (
      <div className="p-4 border rounded-md bg-gray-50">
        <p className="text-sm text-gray-600">No analysis data available for this vendor.</p>
      </div>
    );
  }

  const formatCurrency = (amount: number) => `₹${Math.abs(amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  const getTrendIcon = (direction: string) => {
    if (direction?.toLowerCase().includes('up') || direction?.toLowerCase().includes('positive')) {
      return <TrendingUp className="w-5 h-5 text-green-600" />;
    }
    if (direction?.toLowerCase().includes('down') || direction?.toLowerCase().includes('negative')) {
      return <TrendingDown className="w-5 h-5 text-red-600" />;
    }
    return <Activity className="w-5 h-5 text-gray-600" />;
  };

  // Helper: split insights text into bullet lines and normalize currency
  const toInsightLines = (ins: any): string[] => {
    const raw: string = String(ins || '')
      .replace(/₹/g, '$')
      .replace(/\bRs\.?\s?/gi, '$')
      .replace(/\bINR\s?/gi, '$');
    // Try bullet split like "1. foo 2. bar"
    const bulletParts = raw.split(/\s(?=\d+\.)/g).map(s => s.trim()).filter(Boolean);
    if (bulletParts.length > 1) {
      return bulletParts.map(s => s.replace(/^\d+\.?\s*/, '').replace(/^[-*]\s*/, '').replace(/^\*\*/g, '').replace(/\*\*$/g, ''));
    }
    // Fallback: sentences
    return raw
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim().replace(/^\d+\.?\s*/, '').replace(/^[-*]\s*/, ''))
      .filter(s => s.length > 0);
  };

  return (
    <div className="space-y-6">

      {/* Cash Flow Categories */}
      {vendorData.cash_flow_categories && (
        <div className="p-4 border rounded-md">
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <DollarSign className="w-5 h-5" />
            Cash Flow Categories
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(vendorData.cash_flow_categories).map(([category, amount]) => (
              <div key={category} className="p-3 bg-white border rounded-md">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">{category}</span>
                  <span className={`text-sm font-semibold ${(amount as number) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(amount as number)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}


      {/* Recommendations */}
      {vendorData.recommendations && vendorData.recommendations.length > 0 && (
        <div className="p-4 border rounded-md">
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            Recommendations
          </h4>
          <div className="space-y-2">
            {vendorData.recommendations.slice(0, 5).map((rec: string, idx: number) => (
              <div key={idx} className="p-3 bg-yellow-50 rounded-md">
                <p className="text-sm text-yellow-900 flex items-start gap-2">
                  <span className="text-green-600 mt-1">✓</span>
                  <span>{rec}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Historical Chart */}
      {vendorData.historical_data && vendorData.historical_data.length > 0 && (
        <div className="p-4 border rounded-md">
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <Calendar className="w-5 h-5" />
            Historical Trend
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={vendorData.historical_data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="period" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="amount" stroke="#3b82f6" strokeWidth={2} name="Amount" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default function VendorsPage() {
  const [vendors, setVendors] = useState<Vendor[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [extracting, setExtracting] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [selectedVendor, setSelectedVendor] = useState<string>("");
  const [vendorAnalysis, setVendorAnalysis] = useState<VendorAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string>("");
  const [vendorTxs, setVendorTxs] = useState<Array<any>>([]);
  const [txSummary, setTxSummary] = useState<{inflows:number; outflows:number; net:number; count:number; status?:string; frequency?:string} | null>(null);
  const [vendorReasoning, setVendorReasoning] = useState<any>(null);
  const [loadingVendorReasoning, setLoadingVendorReasoning] = useState(false);
  const [showVendorReasoningModal, setShowVendorReasoningModal] = useState(false);
  const [hasLoaded, setHasLoaded] = useState<boolean>(false);

  useEffect(() => {
    // ✅ Prevent double-fetch in React StrictMode
    if (hasLoaded) {
      console.log(`[VENDORS] Already loaded, skipping duplicate fetch`);
      return;
    }

    let isCancelled = false;
    const abortController = new AbortController();
    
    const fetchVendors = async () => {
      try {
        setLoading(true);
        setError("");
        console.log(`[VENDORS] Fetching from: ${API_BASE}/get-dropdown-data`);
        
        // Use endpoint that exposes vendors from uploaded bank data
        const res = await fetch(`${API_BASE}/get-dropdown-data`, { 
          cache: "no-store",
          signal: abortController.signal,
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          }
        });
        
        // Check if component unmounted
        if (isCancelled) {
          console.log(`[VENDORS] Request cancelled - component unmounted`);
          return;
        }
        
        console.log(`[VENDORS] Response status: ${res.status} ${res.statusText}`);
        
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();
        
        // Check if component unmounted before setting state
        if (isCancelled) {
          console.log(`[VENDORS] Response received but component unmounted`);
          return;
        }
        
        const list: string[] = Array.isArray(data?.vendors) ? data.vendors : [];
        const items: Vendor[] = list.map((v) => ({ name: String(v) }));
        setVendors(items);
        setHasLoaded(true); // Mark as loaded to prevent duplicate fetches
        console.log(`[VENDORS] Vendors loaded successfully: ${items.length} vendors`);
      } catch (e: any) {
        // Don't show error if request was cancelled or connection reset after successful load
        if (isCancelled || (e?.name === 'AbortError') || (e?.message?.includes('Failed to fetch') && hasLoaded)) {
          console.log(`[VENDORS] Request cancelled or duplicate fetch ignored`);
          return;
        }
        console.error(`[VENDORS] Error loading vendors:`, e);
        if (!isCancelled) {
          setError(e?.message || "Failed to load vendors");
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };
    
    fetchVendors();
    
    // Cleanup function to cancel request if component unmounts
    return () => {
      isCancelled = true;
      abortController.abort();
      console.log(`[VENDORS] Cleanup: aborting fetch request`);
    };
  }, [hasLoaded]);

  const runVendorExtraction = async () => {
    try {
      setExtracting(true);
      setError("");
      const res = await fetch(`${API_BASE}/extract-vendors-for-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: "frontend" })
      });
      if (!res.ok) throw new Error(`Extraction failed (${res.status})`);
      // Refresh vendor list
      const vendorsRes = await fetch(`${API_BASE}/get-dropdown-data`, { cache: "no-store" });
      const data = await vendorsRes.json();
      const list: string[] = Array.isArray(data?.vendors) ? data.vendors : [];
      setVendors(list.map((v) => ({ name: String(v) })));
    } catch (e: any) {
      setError(e?.message || "Vendor extraction failed");
    } finally {
      setExtracting(false);
    }
  };

  const analyzeVendor = async (vendorName: string) => {
    try {
      setAnalyzing(true);
      setAnalysisError("");
      setSelectedVendor(vendorName);
      
      const res = await fetch(`${API_BASE}/vendor-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vendor: vendorName })
      });
      
      if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
      const data = await res.json();
      setVendorAnalysis(data);

      // Also fetch concrete transactions for this vendor to display inflow/outflow and table
      try {
        const txRes = await fetch(`${API_BASE}/view_vendor_transactions/${encodeURIComponent(vendorName)}`, { cache: "no-store" });
        if (txRes.ok) {
          const txData = await txRes.json();
          const txs = Array.isArray(txData?.transactions) ? txData.transactions : [];
          setVendorTxs(txs);
          // compute inflow/outflow summary
          const inflows = txs.reduce((s:number, t:any) => s + Number(t.Inward_Amount || t['Inward Amount'] || t.inward_amount || (Number(t.amount) > 0 ? Number(t.amount) : 0) || 0), 0);
          const outflows = txs.reduce((s:number, t:any) => s + Number(t.Outward_Amount || t['Outward Amount'] || t.outward_amount || (Number(t.amount) < 0 ? Math.abs(Number(t.amount)) : 0) || 0), 0);
          setTxSummary({ inflows, outflows, net: inflows - outflows, count: txs.length, status: txData?.summary_cards?.cash_flow_status?.value, frequency: txData?.summary_cards?.payment_patterns?.value });
        } else {
          setVendorTxs([]);
          setTxSummary(null);
        }
      } catch (_) {
        setVendorTxs([]);
        setTxSummary(null);
      }
    } catch (e: any) {
      setAnalysisError(e?.message || "Vendor analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  const fetchVendorReasoning = async (vendorName: string) => {
    setLoadingVendorReasoning(true);
    setShowVendorReasoningModal(true);
    
    try {
      const response = await fetch(`${API_BASE}/ai-reasoning/vendor-landscape`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vendors: [vendorName],
          transactions: vendorTxs
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        // The backend returns the analysis in data.analysis
        let reasoningData = data.analysis;
        
        // If the analysis is a JSON string, parse it
        if (typeof reasoningData === 'string') {
          try {
            // Remove any prefix like ""json from the string
            let jsonString = reasoningData.trim();
            
            // Remove various JSON prefixes
            const prefixesToRemove = [
              '""json ',
              '"json ',
              'json ',
              '```json\n',
              '```json',
              '```\n',
              '```'
            ];
            
            for (const prefix of prefixesToRemove) {
              if (jsonString.startsWith(prefix)) {
                jsonString = jsonString.substring(prefix.length);
                break;
              }
            }
            
            // Clean up any trailing characters that might cause issues
            jsonString = jsonString.trim().replace(/```$/, '').trim();
            
            reasoningData = JSON.parse(jsonString);
          } catch (e) {
            reasoningData = { concentration_analysis: reasoningData };
          }
        }
        
        // If reasoningData is an object but has nested JSON strings, parse them
        if (typeof reasoningData === 'object' && reasoningData !== null) {
          // Check if any field contains markdown-wrapped JSON
          Object.keys(reasoningData).forEach(key => {
            if (typeof reasoningData[key] === 'string' && reasoningData[key].includes('```json')) {
              try {
                const jsonContent = reasoningData[key]
                  .replace(/```json/g, '')
                  .replace(/```/g, '')
                  .trim();
                const parsedContent = JSON.parse(jsonContent);
                // Merge the parsed content with the existing data
                reasoningData = { ...reasoningData, ...parsedContent };
              } catch (e) {
                console.error(`Failed to parse nested JSON in ${key}:`, e);
              }
            }
          });
        }
        
        setVendorReasoning(reasoningData);
      } else {
        setVendorReasoning({ error: 'Failed to fetch vendor reasoning' });
      }
    } catch (e) {
      setVendorReasoning({ error: 'Error fetching vendor reasoning' });
    } finally {
      setLoadingVendorReasoning(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight">Vendors</h1>
          <p className="text-muted-foreground">Aggregated insights per vendor.</p>
        </div>

        <div className="flex items-center gap-3">
          {loading && <div>Loading…</div>}
          {error && <div className="text-red-600">{error}</div>}
        </div>

        <div className="flex items-center gap-3">
          <button
            className="px-3 py-2 border rounded-md text-sm"
            onClick={runVendorExtraction}
            disabled={extracting}
          >
            {extracting ? "Extracting…" : "Extract Vendors"}
          </button>
          
        </div>

        {!loading && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {vendors.map((v, idx) => (
              <div 
                key={idx} 
                className={`border rounded-md p-4 cursor-pointer transition-all hover:shadow-md ${
                  selectedVendor === v.name ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:bg-gray-50'
                }`}
                onClick={() => analyzeVendor(v.name)}
              >
                <div className="font-medium flex items-center gap-2">
                  <Building2 className="w-4 h-4" />
                  {v.name}
                </div>
                {typeof v.count === "number" && (
                  <div className="text-xs text-gray-500 mt-1">{v.count} transactions</div>
                )}
                {(v.inflow || v.outflow) && (
                  <div className="mt-2 text-sm">
                    <div className="flex items-center gap-1">
                      <TrendingUp className="w-3 h-3 text-green-600" />
                      Inflow: {Number(v.inflow || 0).toLocaleString()}
                    </div>
                    <div className="flex items-center gap-1">
                      <TrendingDown className="w-3 h-3 text-red-600" />
                      Outflow: {Number(v.outflow || 0).toLocaleString()}
                    </div>
                  </div>
                )}
                <div className="mt-2 text-xs text-blue-600">
                  Click to analyze →
                </div>
              </div>
            ))}
            {vendors.length === 0 && (
              <div className="text-gray-500">No vendors available yet.</div>
            )}
          </div>
        )}

        {/* Vendor Analysis Results */}
        {selectedVendor && (
          <div className="mt-6">
            <div className="flex items-center gap-3 mb-4">
              <h2 className="text-xl font-semibold">Analysis for {selectedVendor}</h2>
              {analyzing && <div className="text-sm text-blue-600">Analyzing...</div>}
              {analysisError && <div className="text-sm text-red-600">{analysisError}</div>}
            </div>
            
            {vendorAnalysis && (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <VendorAnalysisDisplay analysis={vendorAnalysis} vendorName={selectedVendor} />
                  <button 
                    onClick={() => fetchVendorReasoning(selectedVendor)}
                    className="text-sm bg-blue-100 text-blue-700 px-3 py-2 rounded hover:bg-blue-200"
                    title="View AI Vendor Reasoning"
                  >
                     Details
                  </button>
                </div>
              </div>
            )}

            {/* Inflow/Outflow Summary + Transactions Table */}
            {txSummary && (
              <div className="mt-6 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="p-6 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
                    <div className="text-sm text-gray-600 mb-2">Inflows</div>
                    <div className="text-2xl font-bold text-green-600">₹{txSummary.inflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                  </div>
                  <div className="p-6 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
                    <div className="text-sm text-gray-600 mb-2">Outflows</div>
                    <div className="text-2xl font-bold text-red-600">₹{txSummary.outflows.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                  </div>
                  <div className="p-6 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
                    <div className="text-sm text-gray-600 mb-2">NetFlow</div>
                    <div className={`text-2xl font-bold ${txSummary.net >= 0 ? 'text-green-600' : 'text-red-600'}`}>₹{txSummary.net.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                  </div>
                  <div className="p-6 border rounded-lg bg-white shadow-sm hover:shadow-md transition-shadow">
                    <div className="text-sm text-gray-600 mb-2">Transactions</div>
                    <div className="text-2xl font-bold text-blue-600">{txSummary.count}</div>
                  </div>
                </div>

                {vendorTxs.length > 0 && (
                  <div className="p-4 border rounded-md bg-white overflow-auto">
                    <div className="font-semibold mb-3">Transactions</div>
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-gray-600">
                          <th className="py-2 pr-3">Date</th>
                          <th className="py-2 pr-3">Description</th>
                          <th className="py-2 pr-3 text-right">Inward</th>
                          <th className="py-2 pr-3 text-right">Outward</th>
                          <th className="py-2 pr-3 text-right">Balance</th>
                          <th className="py-2 pr-3">Category</th>
                        </tr>
                      </thead>
                      <tbody>
                        {vendorTxs.map((t:any, i:number) => {
                          // Enhanced date formatting with multiple fallbacks
                          let formattedDate = "";
                          
                          // Try multiple date fields and formats
                          const dateFields = [t.date, t.Date, t['Transaction Date'], t['Txn Date'], t['Value Date']];
                          let rawDate = null;
                          
                          for (const field of dateFields) {
                            if (field && field !== "N/A" && field !== "" && field !== null && field !== undefined && field !== "nan" && field !== "None" && field !== "NaT") {
                              rawDate = field;
                              break;
                            }
                          }
                          
                          if (rawDate) {
                            try {
                              // Handle different date formats
                              let dateObj;
                              
                              // If it's already a formatted string, try to parse it
                              if (typeof rawDate === 'string') {
                                // Try ISO format first
                                if (rawDate.includes('T') || rawDate.includes('-')) {
                                  dateObj = new Date(rawDate);
                                } else {
                                  // Try other common formats
                                  dateObj = new Date(rawDate);
                                }
                              } else {
                                dateObj = new Date(rawDate);
                              }
                              
                              if (!isNaN(dateObj.getTime())) {
                                formattedDate = dateObj.toLocaleDateString('en-US', {
                                  weekday: 'short',
                                  year: 'numeric',
                                  month: 'short',
                                  day: 'numeric'
                                });
                              } else {
                                // If parsing fails, try to extract date parts manually
                                const dateStr = String(rawDate);
                                const dateMatch = dateStr.match(/(\d{4})-(\d{1,2})-(\d{1,2})/);
                                if (dateMatch) {
                                  const [, year, month, day] = dateMatch;
                                  const manualDate = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
                                  if (!isNaN(manualDate.getTime())) {
                                    formattedDate = manualDate.toLocaleDateString('en-US', {
                                      weekday: 'short',
                                      year: 'numeric',
                                      month: 'short',
                                      day: 'numeric'
                                    });
                                  } else {
                                    formattedDate = dateStr;
                                  }
                                } else {
                                  formattedDate = dateStr;
                                }
                              }
                            } catch (error) {
                              console.log('Date parsing error:', error, 'Raw date:', rawDate);
                              formattedDate = String(rawDate);
                            }
                          } else {
                            // If no date found, show a placeholder
                            formattedDate = "Date N/A";
                          }
                          
                          // Final check for NaT values
                          if (formattedDate === "NaT" || formattedDate === "nan" || formattedDate === "None") {
                            formattedDate = "Date N/A";
                          }
                          
                          // Extract inward, outward, and balance amounts
                          const inwardAmount = Number(t.Inward_Amount || t['Inward Amount'] || t.inward_amount || (Number(t.amount) > 0 ? Number(t.amount) : 0) || 0);
                          const outwardAmount = Number(t.Outward_Amount || t['Outward Amount'] || t.outward_amount || (Number(t.amount) < 0 ? Math.abs(Number(t.amount)) : 0) || 0);
                          const balance = Number(t.Closing_Balance || t['Closing Balance'] || t.closing_balance || t.Balance || t.balance || 0);
                          
                          return (
                            <tr key={i} className="border-t">
                              <td className="py-2 pr-3 whitespace-nowrap">{formattedDate}</td>
                              <td className="py-2 pr-3">{t.description || t.Description || ''}</td>
                              <td className="py-2 pr-3 text-right text-green-600">
                                {inwardAmount > 0 ? `₹${inwardAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                              </td>
                              <td className="py-2 pr-3 text-right text-red-600">
                                {outwardAmount > 0 ? `₹${outwardAmount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                              </td>
                              <td className="py-2 pr-3 text-right text-blue-600">
                                {balance > 0 ? `₹${balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—'}
                              </td>
                              <td className="py-2 pr-3 whitespace-nowrap">{t.Category || t.category || 'Uncategorized'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Vendor AI Reasoning Modal */}
        {showVendorReasoningModal && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-end items-center mb-4">
                  <button 
                    onClick={() => setShowVendorReasoningModal(false)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                
                {loadingVendorReasoning ? (
                  <div className="text-center py-8">
                    <div className="text-lg">Analyzing</div>
                    <div className="text-sm text-gray-500">Generating strategic insights for {selectedVendor}</div>
                  </div>
                ) : vendorReasoning ? (
                  <div className="space-y-4">
                    {vendorReasoning.error ? (
                      <div className="text-red-600 p-3 bg-red-50 rounded">
                        Error: {vendorReasoning.error}
                      </div>
                    ) : (
                      <>
                        {vendorReasoning.concentration_analysis && (
                          <div>
                            <h3 className="font-semibold text-blue-700 mb-3 flex items-center gap-2">
                              <span className="text-xl">📊</span>
                              Vendor Concentration Analysis
                            </h3>
                            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                              {typeof vendorReasoning.concentration_analysis === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(vendorReasoning.concentration_analysis).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-blue-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-blue-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-blue-700">{vendorReasoning.concentration_analysis}</div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {vendorReasoning.vendor_segmentation && (
                          <div>
                            <h3 className="font-semibold text-green-700 mb-3 flex items-center gap-2">
                              <span className="text-xl">🎯</span>
                              Vendor Segmentation
                            </h3>
                            <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                              {typeof vendorReasoning.vendor_segmentation === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(vendorReasoning.vendor_segmentation).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-green-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-green-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-green-700">{vendorReasoning.vendor_segmentation}</div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {vendorReasoning.financial_insights && (
                          <div>
                            <h3 className="font-semibold text-yellow-700 mb-3 flex items-center gap-2">
                              <span className="text-xl">💰</span>
                              Financial Insights
                            </h3>
                            <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                              {typeof vendorReasoning.financial_insights === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(vendorReasoning.financial_insights).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-yellow-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-yellow-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-yellow-700">{vendorReasoning.financial_insights}</div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {vendorReasoning.risk_assessment && (
                          <div>
                            <h3 className="font-semibold text-red-700 mb-3 flex items-center gap-2">
                              <span className="text-xl">⚠️</span>
                              Risk Assessment
                            </h3>
                            <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                              {typeof vendorReasoning.risk_assessment === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(vendorReasoning.risk_assessment).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-red-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-red-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-red-700">{vendorReasoning.risk_assessment}</div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {vendorReasoning.strategic_recommendations && (
                          <div>
                            <h3 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                              <span className="text-xl">📋</span>
                              Strategic Recommendations
                            </h3>
                            <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                              {typeof vendorReasoning.strategic_recommendations === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(vendorReasoning.strategic_recommendations).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-purple-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-purple-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-purple-700">{vendorReasoning.strategic_recommendations}</div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {vendorReasoning.statistics && (
                          <div className="mt-4 p-3 bg-gray-100 rounded">
                            <div className="text-sm">
                              <strong>Analysis Summary:</strong> {vendorReasoning.statistics.total_vendors} vendors, {vendorReasoning.statistics.total_transactions} transactions analyzed
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}


