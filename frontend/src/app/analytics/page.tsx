"use client";
import React, { useEffect, useState } from "react";
import { DashboardLayout } from "@/components/layout/dashboard-layout";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle, Calendar, DollarSign } from "lucide-react";

// Component to display trend results with proper visualization
function TrendResultsDisplay({ result }: { result: any }) {
  const [expandedTrend, setExpandedTrend] = useState<string | null>(null);
  const [trendReasoning, setTrendReasoning] = useState<any>(null);
  const [loadingTrendReasoning, setLoadingTrendReasoning] = useState(false);
  const [showTrendReasoningModal, setShowTrendReasoningModal] = useState(false);

  if (!result?.data?.trends_analysis) {
    return (
      <div className="mt-4 p-4 border rounded-md bg-gray-50">
        <p className="text-sm text-gray-600">No trend data available</p>
      </div>
    );
  }

  const trendsData = result.data.trends_analysis;
  const summary = result.data.analysis_summary || {};

  // Extract trend names (excluding _summary)
  const trendNames = Object.keys(trendsData).filter(key => key !== '_summary');

  const formatTrendName = (name: string) => {
    return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  const getTrendIcon = (direction: string) => {
    if (direction?.toLowerCase().includes('up') || direction?.toLowerCase().includes('increas') || direction?.toLowerCase().includes('growth')) {
      return <TrendingUp className="w-5 h-5 text-green-600" />;
    }
    if (direction?.toLowerCase().includes('down') || direction?.toLowerCase().includes('decreas') || direction?.toLowerCase().includes('declin')) {
      return <TrendingDown className="w-5 h-5 text-red-600" />;
    }
    return <AlertCircle className="w-5 h-5 text-gray-600" />;
  };

  const getRiskColor = (risk: string) => {
    const riskLower = risk?.toLowerCase() || '';
    if (riskLower.includes('high')) return 'text-red-600 bg-red-50';
    if (riskLower.includes('medium') || riskLower.includes('moderate')) return 'text-orange-600 bg-orange-50';
    if (riskLower.includes('low')) return 'text-green-600 bg-green-50';
    return 'text-gray-600 bg-gray-50';
  };

  // Nicely render metric values including objects like aging distributions
  const renderMetricValue = (key: string, value: any) => {
    if (value === null || value === undefined) return <span>N/A</span>;
    if (typeof value === 'number') {
      if (key.includes('rate') || key.includes('percentage') || key.includes('probability')) {
        return <span>{(value * 100).toFixed(2)}%</span>;
      }
      if (key.includes('revenue') || key.includes('amount') || key.includes('value') || key.includes('overdue') || key.includes('total')) {
        return <span>${Math.abs(value).toLocaleString()}</span>;
      }
      return <span>{value.toFixed(2)}</span>;
    }
    if (Array.isArray(value)) {
      return <span>{value.join(', ')}</span>;
    }
    if (typeof value === 'object') {
      // Common case: aging distribution map { '0-30': n, '31-60': m, ... }
      const entries = Object.entries(value as Record<string, any>);
      if (entries.length === 0) return <span>N/A</span>;
      return (
        <div className="flex flex-wrap gap-2">
          {entries.map(([k, v]) => (
            <span key={k} className="px-2 py-1 rounded bg-gray-100 text-xs text-gray-700">
              {k}: {typeof v === 'number' ? Math.abs(v).toLocaleString() : String(v)}
            </span>
          ))}
        </div>
      );
    }
    // Fallback to string
    return <span>{String(value)}</span>;
  };

  const fetchTrendReasoning = async (trendType: string) => {
    setLoadingTrendReasoning(true);
    setShowTrendReasoningModal(true);
    
    try {
      // Use the existing trends data for AI analysis instead of requiring transactions
      const trendsData = result?.data?.trends_analysis || {};
      const analysisSummary = result?.data?.analysis_summary || {};
      
      // If no trends data available, provide a fallback
      if (!trendsData || Object.keys(trendsData).length === 0) {
        setTrendReasoning({ 
          error: 'No trends data available for AI analysis. Please run a trends analysis first.' 
        });
        return;
      }
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"}/ai-reasoning/trend-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trend_type: trendType,
          trends_data: trendsData,
          analysis_summary: analysisSummary,
          filters: {
            date_range: "last_6_months",
            analysis_type: "comprehensive"
          }
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Raw backend response:', data);
        
        // The backend returns the analysis in data.analysis
        let reasoningData = data.analysis;
        console.log('Reasoning data type:', typeof reasoningData, 'Value:', reasoningData);
        
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
            console.log('✅ Successfully parsed reasoning JSON');
          } catch (e) {
            console.error('JSON parsing error:', e, 'Raw data:', reasoningData);
            reasoningData = { ai_analysis_methodology: reasoningData };
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
                console.log(`✅ Successfully parsed nested JSON from ${key}`);
              } catch (e) {
                console.error(`Failed to parse nested JSON in ${key}:`, e);
              }
            }
          });
        }
        
        setTrendReasoning(reasoningData);
      } else {
        setTrendReasoning({ error: 'Failed to fetch trend reasoning' });
      }
    } catch (e) {
      setTrendReasoning({ error: 'Error fetching trend reasoning' });
    } finally {
      setLoadingTrendReasoning(false);
    }
  };

  return (
    <div className="mt-6 space-y-4">
      {/* Summary Card */}
      <div className="p-4 border rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-600" />
            Analysis Summary
          </h3>
          <button 
            onClick={() => fetchTrendReasoning('comprehensive_analysis')}
            className="text-sm bg-blue-100 text-blue-700 px-3 py-2 rounded hover:bg-blue-200"
            title="View Details"
          >
            Details
          </button>
        </div>
        <div className="flex justify-center">
          <div className="flex flex-col">
            <span className="text-sm text-gray-600">Total Trends Analyzed</span>
            <span className="text-2xl font-bold text-blue-600">{summary.trends_count || 0}</span>
          </div>
        </div>
      </div>

      {/* Individual Trend Cards */}
      <div className="space-y-3">
        {trendNames.map((trendName) => {
          const trend = trendsData[trendName];
          const isExpanded = expandedTrend === trendName;

          return (
            <div key={trendName} className="border rounded-lg overflow-hidden">
              {/* Trend Header */}
              <div 
                className="p-4 bg-white hover:bg-gray-50 cursor-pointer transition-colors"
                onClick={() => setExpandedTrend(isExpanded ? null : trendName)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 flex-1">
                    {getTrendIcon(trend?.trend_direction || '')}
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900">{formatTrendName(trendName)}</h4>
                      <p className="text-sm text-gray-600 mt-1">
                        {trend?.trend_direction || 'No trend data'} • 
                        Confidence: {trend?.confidence ? `${(trend.confidence * 100).toFixed(0)}%` : 'N/A'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {trend?.risk_level && (
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${getRiskColor(trend.risk_level)}`}>
                        {trend.risk_level}
                      </span>
                    )}
                    <button 
                      onClick={(e) => { e.stopPropagation(); fetchTrendReasoning(trendName); }}
                      className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded hover:bg-blue-200"
                      title={`View details for ${formatTrendName(trendName)}`}
                    >
                      Details
                    </button>
                    <button className="text-gray-400 hover:text-gray-600">
                      {isExpanded ? '▼' : '▶'}
                    </button>
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="p-4 bg-gray-50 border-t space-y-4">
                  {/* Key Metrics */}
                  {trend && (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {trend.pattern_strength && (
                        <div className="bg-white p-3 rounded-md">
                          <p className="text-xs text-gray-500">Pattern Strength</p>
                          <p className="text-sm font-semibold">{trend.pattern_strength}</p>
                        </div>
                      )}
                      {trend.forecast_accuracy && (
                        <div className="bg-white p-3 rounded-md">
                          <p className="text-xs text-gray-500">Forecast Accuracy</p>
                          <p className="text-sm font-semibold">{(trend.forecast_accuracy * 100).toFixed(1)}%</p>
                        </div>
                      )}
                      {trend.data_quality && (
                        <div className="bg-white p-3 rounded-md">
                          <p className="text-xs text-gray-500">Data Quality</p>
                          <p className="text-sm font-semibold">{trend.data_quality}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Insights */}
                  {trend?.insights && trend.insights.length > 0 && (
                    <div className="bg-white p-4 rounded-md">
                      <h5 className="font-semibold mb-2 text-sm">Key Insights</h5>
                      <ul className="space-y-1">
                        {trend.insights.slice(0, 5).map((insight: string, idx: number) => (
                          <li key={idx} className="text-sm text-gray-700 flex items-start gap-2">
                            <span className="text-blue-600 mt-1">•</span>
                            <span>{insight}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Historical Data Chart */}
                  {trend?.historical_data && trend.historical_data.length > 0 && (
                    <div className="bg-white p-4 rounded-md">
                      <h5 className="font-semibold mb-3 text-sm">Historical Trend</h5>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={trend.historical_data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="period" 
                            tick={{ fontSize: 12 }}
                            angle={-45}
                            textAnchor="end"
                            height={60}
                          />
                          <YAxis tick={{ fontSize: 12 }} />
                          <Tooltip />
                          <Legend />
                          <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} name="Actual" />
                          {trend.historical_data[0]?.forecast !== undefined && (
                            <Line type="monotone" dataKey="forecast" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" name="Forecast" />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Recommendations */}
                  {trend?.recommendations && trend.recommendations.length > 0 && (
                    <div className="bg-white p-4 rounded-md">
                      <h5 className="font-semibold mb-2 text-sm">Recommendations</h5>
                      <ul className="space-y-1">
                        {trend.recommendations.slice(0, 3).map((rec: string, idx: number) => (
                          <li key={idx} className="text-sm text-gray-700 flex items-start gap-2">
                            <span className="text-green-600 mt-1">✓</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Formatted Data Display */}
                  {trend && (
                    <div className="bg-white p-4 rounded-md">
                      <h5 className="font-semibold mb-3 text-sm">Detailed Analysis</h5>
                      <div className="space-y-3">
                        {/* Business Metrics */}
                        {trend.business_metrics && (
                          <div>
                            <h6 className="text-sm font-medium text-gray-700 mb-2">Business Metrics</h6>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {Object.entries(trend.business_metrics).map(([key, value]) => (
                                <div key={key} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                                  <span className="text-sm text-gray-600 capitalize">
                                    {key.replace(/_/g, ' ')}
                                  </span>
                                    <span className="text-sm font-semibold text-gray-900 text-right">
                                      {renderMetricValue(key.toLowerCase(), value)}
                                    </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* AI Insights */}
                        {trend.ai_insights && Array.isArray(trend.ai_insights) && trend.ai_insights.length > 0 && (
                          <div>
                            <h6 className="text-sm font-medium text-gray-700 mb-2">AI Insights</h6>
                            <div className="space-y-2">
                              {trend.ai_insights.map((insight: string, idx: number) => (
                                <div key={idx} className="p-3 bg-blue-50 rounded-md">
                                  <p className="text-sm text-blue-900">{insight}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Technical Details */}
                        {(trend.technical_details || trend.model_performance) && (
                          <div>
                            <h6 className="text-sm font-medium text-gray-700 mb-2">Technical Details</h6>
                            <div className="space-y-2">
                              {trend.technical_details && (
                                <div className="p-3 bg-gray-50 rounded-md">
                                  <p className="text-sm text-gray-700">
                                    <span className="font-medium">Method:</span> {trend.technical_details.method || 'N/A'}
                                  </p>
                                  {trend.technical_details.accuracy && (
                                    <p className="text-sm text-gray-700">
                                      <span className="font-medium">Accuracy:</span> {(trend.technical_details.accuracy * 100).toFixed(1)}%
                                    </p>
                                  )}
                                </div>
                              )}
                              {trend.model_performance && (
                                <div className="p-3 bg-gray-50 rounded-md">
                                  <p className="text-sm text-gray-700">
                                    <span className="font-medium">Model Performance:</span> {trend.model_performance}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Additional Data Points */}
                        {trend.seasonality && (
                          <div className="p-3 bg-green-50 rounded-md">
                            <h6 className="text-sm font-medium text-green-700 mb-1">Seasonality Analysis</h6>
                            <p className="text-sm text-green-800">{trend.seasonality}</p>
                          </div>
                        )}

                        {trend.anomaly_detection && (
                          <div className="p-3 bg-yellow-50 rounded-md">
                            <h6 className="text-sm font-medium text-yellow-700 mb-1">Anomaly Detection</h6>
                            <p className="text-sm text-yellow-800">{trend.anomaly_detection}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Empty State */}
      {trendNames.length === 0 && (
        <div className="p-8 text-center border rounded-lg bg-gray-50">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-3" />
          <p className="text-gray-600">No trend analysis results available</p>
        </div>
      )}

      {/* AI Trend Reasoning Modal */}
      {showTrendReasoningModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-end items-center mb-4">
                <button 
                  onClick={() => setShowTrendReasoningModal(false)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              
              {loadingTrendReasoning ? (
                <div className="text-center py-8">
                  <div className="text-lg">Analyzing</div>
                  <div className="text-sm text-gray-500">Generating comprehensive trend insights</div>
                </div>
              ) : trendReasoning ? (
                <div className="space-y-4">
                  {trendReasoning.error ? (
                    <div className="text-red-600 p-3 bg-red-50 rounded">
                      Error: {trendReasoning.error}
                    </div>
                  ) : (
                    <>
                      {(trendReasoning.ai_analysis_methodology || trendReasoning.AI_ANALYSIS_METHODOLOGY) && (
                        <div>
                          <h3 className="font-semibold text-blue-700 mb-3 flex items-center gap-2">
                            Interpretation Model
                          </h3>
                          <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            {typeof (trendReasoning.ai_analysis_methodology || trendReasoning.AI_ANALYSIS_METHODOLOGY) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.ai_analysis_methodology || trendReasoning.AI_ANALYSIS_METHODOLOGY).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-blue-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-blue-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-blue-700">{trendReasoning.ai_analysis_methodology || trendReasoning.AI_ANALYSIS_METHODOLOGY}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {(trendReasoning.trend_identification || trendReasoning.TREND_IDENTIFICATION) && (
                        <div>
                          <h3 className="font-semibold text-green-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">📈</span>
                            Trend Identification
                          </h3>
                          <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                            {typeof (trendReasoning.trend_identification || trendReasoning.TREND_IDENTIFICATION) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.trend_identification || trendReasoning.TREND_IDENTIFICATION).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-green-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-green-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-green-700">{trendReasoning.trend_identification || trendReasoning.TREND_IDENTIFICATION}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {(trendReasoning.business_insights || trendReasoning.BUSINESS_INSIGHTS) && (
                        <div>
                          <h3 className="font-semibold text-yellow-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">💼</span>
                            Business Insights
                          </h3>
                          <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                            {typeof (trendReasoning.business_insights || trendReasoning.BUSINESS_INSIGHTS) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.business_insights || trendReasoning.BUSINESS_INSIGHTS).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-yellow-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-yellow-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-yellow-700">{trendReasoning.business_insights || trendReasoning.BUSINESS_INSIGHTS}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {(trendReasoning.predictive_analysis || trendReasoning.PREDICTIVE_ANALYSIS) && (
                        <div>
                          <h3 className="font-semibold text-purple-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">🔮</span>
                            Predictive Analysis
                          </h3>
                          <div className="bg-purple-50 p-4 rounded-lg border border-purple-200">
                            {typeof (trendReasoning.predictive_analysis || trendReasoning.PREDICTIVE_ANALYSIS) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.predictive_analysis || trendReasoning.PREDICTIVE_ANALYSIS).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-purple-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-purple-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-purple-700">{trendReasoning.predictive_analysis || trendReasoning.PREDICTIVE_ANALYSIS}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {(trendReasoning.root_cause_analysis || trendReasoning.ROOT_CAUSE_ANALYSIS) && (
                        <div>
                          <h3 className="font-semibold text-orange-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">🔍</span>
                            Root Cause Analysis
                          </h3>
                            <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
                              {typeof (trendReasoning.root_cause_analysis || trendReasoning.ROOT_CAUSE_ANALYSIS) === 'object' ? (
                                <div className="space-y-3">
                                  {Object.entries(trendReasoning.root_cause_analysis || trendReasoning.ROOT_CAUSE_ANALYSIS).map(([key, value]: [string, any]) => (
                                    <div key={key}>
                                      <div className="font-medium text-orange-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                      <div className="text-sm text-orange-700">{String(value)}</div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="text-sm text-orange-700">{trendReasoning.root_cause_analysis || trendReasoning.ROOT_CAUSE_ANALYSIS}</div>
                              )}
                            </div>
                        </div>
                      )}
                      
                      {(trendReasoning.actionable_recommendations || trendReasoning.ACTIONABLE_RECOMMENDATIONS) && (
                        <div>
                          <h3 className="font-semibold text-red-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">📋</span>
                            Actionable Recommendations
                          </h3>
                          <div className="bg-red-50 p-4 rounded-lg border border-red-200">
                            {typeof (trendReasoning.actionable_recommendations || trendReasoning.ACTIONABLE_RECOMMENDATIONS) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.actionable_recommendations || trendReasoning.ACTIONABLE_RECOMMENDATIONS).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-red-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-red-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-red-700">{trendReasoning.actionable_recommendations || trendReasoning.ACTIONABLE_RECOMMENDATIONS}</div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      {(trendReasoning.risk_opportunity_assessment || trendReasoning.RISK_OPPORTUNITY_ASSESSMENT) && (
                        <div>
                          <h3 className="font-semibold text-indigo-700 mb-3 flex items-center gap-2">
                            <span className="text-xl">⚠️</span>
                            Risk & Opportunity Assessment
                          </h3>
                          <div className="bg-indigo-50 p-4 rounded-lg border border-indigo-200">
                            {typeof (trendReasoning.risk_opportunity_assessment || trendReasoning.RISK_OPPORTUNITY_ASSESSMENT) === 'object' ? (
                              <div className="space-y-3">
                                {Object.entries(trendReasoning.risk_opportunity_assessment || trendReasoning.RISK_OPPORTUNITY_ASSESSMENT).map(([key, value]: [string, any]) => (
                                  <div key={key}>
                                    <div className="font-medium text-indigo-800 mb-1 capitalize">{key.replace(/_/g, ' ')}:</div>
                                    <div className="text-sm text-indigo-700">{String(value)}</div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-sm text-indigo-700">{trendReasoning.risk_opportunity_assessment || trendReasoning.RISK_OPPORTUNITY_ASSESSMENT}</div>
                            )}
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
  );
}

function TrendsClient() {
  "use client";
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
  const [loading, setLoading] = useState<boolean>(false);
  const [types, setTypes] = useState<string[]>([]);
  const [activeTrend, setActiveTrend] = useState<string | null>(null);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    const load = async () => {
      try {
        setError("");
        const r = await fetch(`${API_BASE}/get-available-trend-types`, { cache: "no-store" });
        const j = await r.json();
        const list: string[] = Array.isArray(j?.trend_types) ? j.trend_types : Array.isArray(j) ? j : [];
        setTypes(list);
      } catch (e: any) {
        setError(e?.message || "Failed to load trend types");
      }
    };
    load();
  }, [API_BASE]);

  const formatTrendName = (name: string) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const run = async (requestedTrends: string[]) => {
    if (!requestedTrends || requestedTrends.length === 0) {
      setError("Select at least one trend type.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      // Validate selection first
      const v = await fetch(`${API_BASE}/validate-trend-selection`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_trends: requestedTrends })
      });
      if (!v.ok) throw new Error(`Validate failed (${v.status})`);

      // Run analysis - FIXED: Use 'analysis_type' instead of 'selected_trends'
      const r = await fetch(`${API_BASE}/run-dynamic-trends-analysis`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis_type: requestedTrends })
      });
      if (!r.ok) throw new Error(`Run failed (${r.status})`);
      const j = await r.json();
      setResult(j);
    } catch (e: any) {
      setError(e?.message || "Trend run failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-xl font-semibold">Trends & Forecast</h2>
      {error && <div className="text-red-600 text-sm">{error}</div>}

      {/* Individual Options Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {types.length > 1 && (
          <div
            className={`p-4 border rounded-md cursor-pointer transition-all hover:shadow ${
              activeTrend === "__all__" ? "ring-2 ring-blue-500 bg-blue-50" : "hover:bg-gray-50"
            }`}
            onClick={() => {
              setActiveTrend("__all__");
              run(types);
            }}
          >
            <div className="font-medium text-gray-900">All Trends</div>
            <p className="text-xs text-blue-600 mt-1">
              {loading && activeTrend === "__all__" ? "Running…" : `Run all ${types.length} analyses`}
            </p>
          </div>
        )}
        {types.map((t) => (
          <div
            key={t}
            className={`p-4 border rounded-md cursor-pointer transition-all hover:shadow ${
              activeTrend === t ? "ring-2 ring-blue-500 bg-blue-50" : "hover:bg-gray-50"
            }`}
            onClick={() => {
              setActiveTrend(t);
              run([t]);
            }}
          >
            <div className="font-medium text-gray-900">{formatTrendName(t)}</div>
            <p className="text-xs text-blue-600 mt-1">
              {loading && activeTrend === t ? "Running analysis…" : "Click to run analysis"}
            </p>
          </div>
        ))}
        {types.length === 0 && <div className="text-sm text-gray-500">No trend types available.</div>}
      </div>

      {result && <TrendResultsDisplay result={result} />}
    </div>
  );
}

export default function AnalyticsPage() {
  return (
    <DashboardLayout>
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight">Analytics</h1>
          <p className="text-muted-foreground">Trends and forecasting analysis.</p>
        </div>
        <TrendsClient />
      </div>
    </DashboardLayout>
  );
}


