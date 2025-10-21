"use client";

import { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/dashboard-layout";

type Transaction = {
  date: string;
  description: string;
  amount: number;
  category?: string;
  vendor?: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function TransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [filter, setFilter] = useState<'All' | 'Operating Activities' | 'Investing Activities' | 'Financing Activities'>('All');
  const [insights, setInsights] = useState<string[]>([]);
  const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);
  const [reasoning, setReasoning] = useState<any>(null);
  const [loadingReasoning, setLoadingReasoning] = useState(false);
  const [showReasoningModal, setShowReasoningModal] = useState(false);

  useEffect(() => {
    const fetchTransactions = async () => {
      try {
        setLoading(true);
        setError("");
        // Use endpoint that returns uploaded bank data without requiring reconciliation
        const res = await fetch(`${API_BASE}/get-current-data`, { cache: "no-store" });
        if (!res.ok) throw new Error(`API ${res.status}`);
        const data = await res.json();

        const rows: any[] = Array.isArray(data?.bank_data) ? data.bank_data : [];
        const items: Transaction[] = rows.map((r) => {
          const dateVal = r.Date || r["Transaction Date"] || r.date || "";
          const descVal = r.Description || r["Transaction Description"] || r.description || "";
          const amtVal = Number(
            r.Amount ?? r["Credit Amount"] ?? r["Debit Amount"] ?? r.amount ?? 0
          );
          const catRaw = String(r.Category || r.category || '').trim();
          const vendorVal = r.Vendor || r.vendor;
          
          // Improved date formatting
          let formattedDate = "";
          if (dateVal && dateVal !== "N/A" && dateVal !== "") {
            try {
              // Try to parse and format the date
              const dateObj = new Date(dateVal);
              if (!isNaN(dateObj.getTime())) {
                formattedDate = dateObj.toLocaleDateString('en-US', {
                  weekday: 'short',
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric'
                });
              } else {
                formattedDate = String(dateVal);
              }
            } catch {
              formattedDate = String(dateVal);
            }
          }
          
          // Enhanced category mapping with better logic
          const lc = catRaw.toLowerCase();
          const descLc = String(descVal).toLowerCase();
          let topCat = catRaw;
          
          // Skip if already properly categorized
          if (catRaw && catRaw !== 'Uncategorized' && !lc.includes('uncategoriz')) {
            topCat = catRaw;
          } else {
            // Enhanced categorization logic
            if (/(energy|power|electricity|generation|mwh|kwh|discom|ntpc|thermal|solar|wind)/.test(descLc)) {
              topCat = 'Operating Activities';
            } else if (/(salary|wages|payroll|staff|employee|benefits)/.test(descLc)) {
              topCat = 'Operating Activities';
            } else if (/(maintenance|repair|service|inspection|spares|parts)/.test(descLc)) {
              topCat = 'Operating Activities';
            } else if (/(coal|fuel|procurement|shipping|freight|logistics|transport)/.test(descLc)) {
              topCat = 'Operating Activities';
            } else if (/(equipment|plant|machinery|construction|contractor|project|capex|turbine|boiler)/.test(descLc)) {
              topCat = 'Investing Activities';
            } else if (/(loan|interest|equity|dividend|debt|bank|credit|financing)/.test(descLc)) {
              topCat = 'Financing Activities';
            } else if (/(capital|investment|acquisition|purchase|upgrade)/.test(descLc)) {
              topCat = 'Investing Activities';
            } else {
              topCat = 'Operating Activities'; // Default to Operating instead of Uncategorized
            }
          }
          
          return {
            date: formattedDate,
            description: String(descVal),
            amount: amtVal,
            category: topCat,
            vendor: vendorVal,
          } as Transaction;
        });
        setTransactions(items);
      } catch (e: any) {
        setError(e?.message || "Failed to load transactions");
      } finally {
        setLoading(false);
      }
    };
    fetchTransactions();
  }, []);

  // Fetch business insights from backend based on current filter (Option A)
  useEffect(() => {
    const fetchInsights = async () => {
      try {
        // Only call backend for specific categories; skip when All to avoid ambiguity
        const txType = filter === 'All' ? 'Operating Activities' : filter;
        const res = await fetch(`${API_BASE}/transaction-analysis`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ transaction_type: txType, analysis_type: 'cash_flow', performance_mode: 'fast' })
        });
        if (!res.ok) throw new Error('analysis failed');
        const data = await res.json();
        const text = JSON.stringify(data?.reasoning_explanations || data?.data || '');
        // naive split into bullet lines
        const lines = String(text).split(/(?<=[.!?])\s+/).map(s => s.replace(/[\r\n]+/g, ' ').trim()).filter(Boolean).slice(0, 6);
        setInsights(lines);
      } catch {
        setInsights([]);
      }
    };
    fetchInsights();
  }, [filter]);

  const fetchTransactionReasoning = async (transaction: Transaction) => {
    setLoadingReasoning(true);
    setSelectedTransaction(transaction);
    setShowReasoningModal(true);
    
    try {
      const response = await fetch(`${API_BASE}/ai-reasoning/categorization`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_description: transaction.description,
          category: transaction.category,
          all_transactions: transactions.slice(0, 10) // Include some context
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Raw backend response:', data);
        
        // The backend returns the reasoning in data.reasoning
        let reasoningData = data.reasoning;
        console.log('Reasoning data type:', typeof reasoningData, 'Value:', reasoningData);
        
        // If the reasoning is a JSON string, parse it
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
            console.log('✅ Successfully parsed reasoning JSON');
          } catch (e) {
            console.error('JSON parsing error:', e, 'Raw data:', reasoningData);
            // If parsing fails, treat as plain text
            reasoningData = { reasoning_process: reasoningData };
          }
        }
        
        // If reasoningData is an object but has nested JSON strings, parse them
        if (typeof reasoningData === 'object' && reasoningData !== null) {
          // Check if reasoning_process is a string with markdown JSON
          if (reasoningData.reasoning_process && typeof reasoningData.reasoning_process === 'string') {
            if (reasoningData.reasoning_process.includes('```json')) {
              try {
                const jsonContent = reasoningData.reasoning_process
                  .replace(/```json/g, '')
                  .replace(/```/g, '')
                  .trim();
                const parsedReasoning = JSON.parse(jsonContent);
                // Merge the parsed content with the existing data
                reasoningData = { ...reasoningData, ...parsedReasoning };
                console.log('✅ Successfully parsed nested JSON from reasoning_process');
              } catch (e) {
                console.error('Failed to parse nested JSON in reasoning_process:', e);
              }
            }
          }
        }
        
        setReasoning(reasoningData);
      } else {
        setReasoning({ error: 'Failed to fetch reasoning' });
      }
    } catch (e) {
      setReasoning({ error: 'Error fetching reasoning' });
    } finally {
      setLoadingReasoning(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight">Transactions</h1>
          <p className="text-muted-foreground">Review parsed rows from the last upload.</p>
        </div>

      {loading && <div>Loading…</div>}
      {error && <div className="text-red-600">{error}</div>}

      {!loading && !error && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            {(['All','Operating Activities','Investing Activities','Financing Activities'] as const).map(opt => (
              <button
                key={opt}
                className={`px-3 py-1 border rounded-md text-sm ${filter===opt ? 'bg-blue-600 text-white' : 'bg-white'}`}
                onClick={() => setFilter(opt)}
              >
                {opt}
              </button>
            ))}
          </div>

          {/* Summary cards */}
          {(() => {
            const filtered = filter==='All' ? transactions : transactions.filter(t => String(t.category).toLowerCase().startsWith(filter.split(' ')[0].toLowerCase()));
            const inflows = filtered.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);
            const outflows = filtered.filter(t => t.amount < 0).reduce((s, t) => s + Math.abs(t.amount), 0);
            const net = inflows - outflows;
            const count = filtered.length;
            return (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-3 border rounded-md bg-white">
                  <div className="text-sm text-gray-600">Inflows</div>
                  <div className="text-lg font-semibold text-green-600">${Math.abs(inflows).toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-white">
                  <div className="text-sm text-gray-600">Outflows</div>
                  <div className="text-lg font-semibold text-red-600">${Math.abs(outflows).toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-white">
                  <div className="text-sm text-gray-600">NetFlow</div>
                  <div className={`text-lg font-semibold ${net >= 0 ? 'text-green-600' : 'text-red-600'}`}>${Math.abs(net).toLocaleString()}</div>
                </div>
                <div className="p-3 border rounded-md bg-white">
                  <div className="text-sm text-gray-600">Transactions</div>
                  <div className="text-lg font-semibold">{count}</div>
                </div>
              </div>
            );
          })()}

          {/* Business Insights from backend (Option A) */}
          {insights.length > 0 && (
            <div className="p-4 border rounded-md">
              <div className="font-semibold mb-2">Business Insights ({filter})</div>
              <div className="space-y-2">
                {insights.map((line, i) => (
                  <div key={i} className="p-2 bg-green-50 rounded-md text-sm text-green-900">{line.replace(/₹/g, '$')}</div>
                ))}
              </div>
            </div>
          )}

          <div className="overflow-x-auto border rounded-md">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left">Date</th>
                  <th className="px-3 py-2 text-left">Description</th>
                  <th className="px-3 py-2 text-right">Amount</th>
                  <th className="px-3 py-2 text-left">Category</th>
                </tr>
              </thead>
              <tbody>
                {(filter==='All' ? transactions : transactions.filter(t => String(t.category).toLowerCase().startsWith(filter.split(' ')[0].toLowerCase()))).map((t, idx) => (
                  <tr key={idx} className="border-t hover:bg-gray-50">
                    <td className="px-3 py-2 whitespace-nowrap">{t.date}</td>
                    <td className="px-3 py-2">{t.description}</td>
                    <td className={`px-3 py-2 text-right ${t.amount>=0?'text-green-600':'text-red-600'}`}>${Math.abs(Number(t.amount)).toLocaleString()}</td>
                    <td className="px-3 py-2 flex items-center gap-2">
                      <span>{t.category || "—"}</span>
                      <button 
                        onClick={() => fetchTransactionReasoning(t)}
                        className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded hover:bg-blue-200"
                        title="View AI Reasoning"
                      >
                        🤖 Details
                      </button>
                    </td>
                  </tr>
                ))}
                {transactions.length === 0 && (
                  <tr>
                    <td className="px-3 py-6 text-center text-gray-500" colSpan={4}>
                      No transactions available. Try uploading a file on Upload Data.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* AI Reasoning Modal */}
      {showReasoningModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-semibold">🤖 AI Reasoning</h2>
                <button 
                  onClick={() => setShowReasoningModal(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              
              {selectedTransaction && (
                <div className="mb-4 p-3 bg-gray-50 rounded">
                  <div className="font-medium">Transaction:</div>
                  <div className="text-sm text-gray-600">{selectedTransaction.description}</div>
                  <div className="text-sm">Amount: ${Math.abs(Number(selectedTransaction.amount)).toLocaleString()}</div>
                  <div className="text-sm">Category: {selectedTransaction.category}</div>
                </div>
              )}
              
              {loadingReasoning ? (
                <div className="text-center py-8">
                  <div className="text-lg">🤖 AI is analyzing...</div>
                  <div className="text-sm text-gray-500">Generating reasoning for this transaction</div>
                </div>
              ) : reasoning ? (
                <div className="space-y-4">
                  {reasoning.error ? (
                    <div className="text-red-600 p-3 bg-red-50 rounded">
                      Error: {reasoning.error}
                    </div>
                  ) : (
                    <>
                      {reasoning.reasoning_process && (
                        <div>
                          <h3 className="font-semibold text-blue-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">🧠</span>
                            AI Reasoning Process
                          </h3>
                          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            {typeof reasoning.reasoning_process === 'object' ? (
                              <div className="space-y-3">
                                {reasoning.reasoning_process.analysis && (
                                  <div>
                                    <div className="font-medium text-blue-800 mb-1">Analysis:</div>
                                    <div className="text-sm text-blue-700">{reasoning.reasoning_process.analysis}</div>
                                  </div>
                                )}
                                {reasoning.reasoning_process.keywords_patterns && (
                                  <div>
                                    <div className="font-medium text-blue-800 mb-1">Key Patterns:</div>
                                    <div className="flex flex-wrap gap-1">
                                      {(Array.isArray(reasoning.reasoning_process.keywords_patterns) 
                                        ? reasoning.reasoning_process.keywords_patterns 
                                        : [reasoning.reasoning_process.keywords_patterns]
                                      ).map((pattern: string, i: number) => (
                                        <span key={i} className="bg-blue-200 text-blue-800 px-2 py-1 rounded text-xs">
                                          {pattern}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {reasoning.reasoning_process.category_choice && (
                                  <div>
                                    <div className="font-medium text-blue-800 mb-1">Category Choice:</div>
                                    <div className="text-sm text-blue-700">{reasoning.reasoning_process.category_choice}</div>
                                  </div>
                                )}
                                {reasoning.reasoning_process.rationale && (
                                  <div>
                                    <div className="font-medium text-blue-800 mb-1">Rationale:</div>
                                    <div className="text-sm text-blue-700">{reasoning.reasoning_process.rationale}</div>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-sm text-blue-700">{reasoning.reasoning_process}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {reasoning.business_impact && (
                        <div>
                          <h3 className="font-semibold text-green-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">💼</span>
                            Business Impact
                          </h3>
                          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                            {typeof reasoning.business_impact === 'object' ? (
                              <div className="space-y-3">
                                {reasoning.business_impact.cash_flow_statements && (
                                  <div>
                                    <div className="font-medium text-green-800 mb-1">Cash Flow Impact:</div>
                                    <div className="text-sm text-green-700">{reasoning.business_impact.cash_flow_statements}</div>
                                  </div>
                                )}
                                {reasoning.business_impact.impact_on_activities && (
                                  <div>
                                    <div className="font-medium text-green-800 mb-1">Activity Impact:</div>
                                    <div className="space-y-2">
                                      {typeof reasoning.business_impact.impact_on_activities === 'object' ? (
                                        Object.entries(reasoning.business_impact.impact_on_activities).map(([activity, impact]: [string, any]) => (
                                          <div key={activity} className="flex items-start gap-2">
                                            <span className="font-medium text-green-800 capitalize min-w-[80px]">{activity}:</span>
                                            <span className="text-sm text-green-700">{impact}</span>
                                          </div>
                                        ))
                                      ) : (
                                        <div className="text-sm text-green-700">{reasoning.business_impact.impact_on_activities}</div>
                                      )}
                                    </div>
                                  </div>
                                )}
                                {reasoning.business_impact.financial_reporting && (
                                  <div>
                                    <div className="font-medium text-green-800 mb-1">Financial Reporting:</div>
                                    <div className="text-sm text-green-700">{reasoning.business_impact.financial_reporting}</div>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-sm text-green-700">{reasoning.business_impact}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {reasoning.validation && (
                        <div>
                          <h3 className="font-semibold text-yellow-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">✅</span>
                            Validation & Quality Check
                          </h3>
                          <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                            {typeof reasoning.validation === 'object' ? (
                              <div className="space-y-3">
                                {reasoning.validation.categorization_correctness && (
                                  <div className="flex items-center gap-2">
                                    <span className="font-medium text-yellow-800">Correctness:</span>
                                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                                      reasoning.validation.categorization_correctness.toLowerCase().includes('yes') 
                                        ? 'bg-green-100 text-green-800' 
                                        : 'bg-red-100 text-red-800'
                                    }`}>
                                      {reasoning.validation.categorization_correctness}
                                    </span>
                                  </div>
                                )}
                                {reasoning.validation.red_flags && (
                                  <div>
                                    <div className="font-medium text-yellow-800 mb-1">⚠️ Considerations:</div>
                                    <div className="text-sm text-yellow-700">{reasoning.validation.red_flags}</div>
                                  </div>
                                )}
                                {reasoning.validation.alternative_interpretations && (
                                  <div>
                                    <div className="font-medium text-yellow-800 mb-1">Alternative Views:</div>
                                    <div className="text-sm text-yellow-700">{reasoning.validation.alternative_interpretations}</div>
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div className="text-sm text-yellow-700">{reasoning.validation}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {reasoning.recommendations && (
                        <div>
                          <h3 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">📋</span>
                            Recommendations & Next Steps
                          </h3>
                          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                            {typeof reasoning.recommendations === 'object' ? (
                              <div className="space-y-3">
                                {reasoning.recommendations.action && (
                                  <div>
                                    <div className="font-medium text-purple-800 mb-1">🎯 Action Required:</div>
                                    <div className="text-sm text-purple-700">{reasoning.recommendations.action}</div>
                                  </div>
                                )}
                                {reasoning.recommendations.monitoring && (
                                  <div>
                                    <div className="font-medium text-purple-800 mb-1">👁️ Monitoring:</div>
                                    <div className="text-sm text-purple-700">{reasoning.recommendations.monitoring}</div>
                                  </div>
                                )}
                                {/* Handle case where recommendations might have additional properties */}
                                {Object.keys(reasoning.recommendations).filter(key => 
                                  !['action', 'monitoring'].includes(key)
                                ).map(key => (
                                  <div key={key}>
                                    <div className="font-medium text-purple-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-purple-700">
                                      {typeof reasoning.recommendations[key] === 'string' 
                                        ? reasoning.recommendations[key] 
                                        : JSON.stringify(reasoning.recommendations[key])
                                      }
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-purple-700">{reasoning.recommendations}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {reasoning.confidence_score && (
                        <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                          <div className="flex items-center justify-between">
                            <div className="font-medium text-gray-800">AI Confidence Level:</div>
                            <div className="flex items-center gap-2">
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full transition-all ${
                                    reasoning.confidence_score >= 0.8 ? 'bg-green-500' :
                                    reasoning.confidence_score >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${reasoning.confidence_score * 100}%` }}
                                ></div>
                              </div>
                              <span className={`font-bold text-sm ${
                                reasoning.confidence_score >= 0.8 ? 'text-green-600' :
                                reasoning.confidence_score >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                              }`}>
                                {(reasoning.confidence_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                          <div className="text-xs text-gray-600 mt-1">
                            {reasoning.confidence_score >= 0.8 ? 'High confidence' :
                             reasoning.confidence_score >= 0.6 ? 'Medium confidence' : 'Low confidence'}
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


