'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useDebuggerStore } from '@/store/debugger';
import { PerformanceMetrics } from '@/types';
import { 
  Activity, 
  Clock, 
  Database, 
  Zap,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';

export function PerformancePanel() {
  const [selectedMetric, setSelectedMetric] = useState<keyof PerformanceMetrics>('inference_time_ms');
  const chartRef = useRef<SVGSVGElement>(null);
  const [chartDimensions, setChartDimensions] = useState({ width: 800, height: 300 });
  
  const { performanceMetrics } = useDebuggerStore();

  // Get recent metrics for display
  const recentMetrics = performanceMetrics.slice(-100); // Last 100 data points

  // Resize observer for chart
  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setChartDimensions({ width: width - 40, height: height - 40 });
      }
    });

    const container = chartRef.current?.parentElement;
    if (container) {
      resizeObserver.observe(container);
    }

    return () => resizeObserver.disconnect();
  }, []);

  // Draw time series chart
  useEffect(() => {
    if (!chartRef.current || recentMetrics.length === 0) return;

    const svg = d3.select(chartRef.current);
    const { width, height } = chartDimensions;
    
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(recentMetrics, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(recentMetrics, d => d[selectedMetric] as number) as [number, number])
      .nice()
      .range([innerHeight, 0]);

    // Create line generator
    const line = d3.line<PerformanceMetrics>()
      .x(d => xScale(new Date(d.timestamp)))
      .y(d => yScale(d[selectedMetric] as number))
      .curve(d3.curveMonotoneX);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat((d) => d3.timeFormat('%H:%M:%S')(d as Date)))
      .selectAll('text')
      .style('fill', '#969696')
      .style('font-size', '12px');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('fill', '#969696')
      .style('font-size', '12px');

    // Add grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0, ${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat(() => '')
      )
      .selectAll('line')
      .style('stroke', '#3c3c3c')
      .style('stroke-width', 0.5);

    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat(() => '')
      )
      .selectAll('line')
      .style('stroke', '#3c3c3c')
      .style('stroke-width', 0.5);

    // Add the line
    g.append('path')
      .datum(recentMetrics)
      .attr('fill', 'none')
      .attr('stroke', '#0078d4')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Add dots
    g.selectAll('.dot')
      .data(recentMetrics.slice(-20)) // Only show dots for last 20 points
      .enter().append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(new Date(d.timestamp)))
      .attr('cy', d => yScale(d[selectedMetric] as number))
      .attr('r', 3)
      .attr('fill', '#0078d4');

    // Add tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', '#2d2d2d')
      .style('border', '1px solid #3c3c3c')
      .style('border-radius', '6px')
      .style('padding', '8px')
      .style('font-size', '12px')
      .style('color', '#cccccc')
      .style('pointer-events', 'none');

    g.selectAll('.dot')
      .on('mouseover', function(event, d: any) {
        tooltip.transition().duration(200).style('opacity', .9);
        tooltip.html(`
          <div><strong>${selectedMetric.replace(/_/g, ' ')}</strong></div>
          <div>${(d[selectedMetric] as number).toFixed(2)}</div>
          <div>${new Date(d.timestamp).toLocaleTimeString()}</div>
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        tooltip.transition().duration(500).style('opacity', 0);
      });

    // Cleanup function
    return () => {
      d3.select('body').selectAll('.tooltip').remove();
    };

  }, [recentMetrics, selectedMetric, chartDimensions]);

  const getMetricInfo = (metric: keyof PerformanceMetrics) => {
    const recent = recentMetrics.slice(-10);
    if (recent.length === 0) return null;

    const current = recent[recent.length - 1][metric] as number;
    const previous = recent.length > 1 ? recent[recent.length - 2][metric] as number : current;
    const avg = recent.reduce((sum, m) => sum + (m[metric] as number), 0) / recent.length;
    const trend = current > previous ? 'up' : current < previous ? 'down' : 'stable';

    return { current, avg, trend };
  };

  const metricOptions = [
    { key: 'inference_time_ms' as keyof PerformanceMetrics, label: 'Inference Time (ms)', icon: Clock },
    { key: 'routing_overhead_ms' as keyof PerformanceMetrics, label: 'Routing Overhead (ms)', icon: Zap },
    { key: 'memory_usage_mb' as keyof PerformanceMetrics, label: 'Memory Usage (MB)', icon: Database },
    { key: 'cache_hit_rate' as keyof PerformanceMetrics, label: 'Cache Hit Rate', icon: Activity },
    { key: 'throughput_tokens_per_sec' as keyof PerformanceMetrics, label: 'Throughput (tok/s)', icon: TrendingUp },
  ];

  const latestMetric = recentMetrics.length > 0 ? recentMetrics[recentMetrics.length - 1] : null;

  return (
    <div className="flex flex-col h-full bg-devtools-background">
      {/* Header */}
      <div className="bg-devtools-surface border-b border-devtools-border p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity size={16} />
            <span className="font-semibold text-devtools-text">Performance</span>
          </div>
          
          <div className="text-sm text-devtools-textSecondary">
            {recentMetrics.length} data points
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-5 gap-4 p-4 bg-devtools-surface border-b border-devtools-border">
        {metricOptions.map((option) => {
          const Icon = option.icon;
          const info = getMetricInfo(option.key);
          const isSelected = selectedMetric === option.key;
          
          return (
            <button
              key={option.key}
              onClick={() => setSelectedMetric(option.key)}
              className={`metric-card text-left transition-all duration-200 ${
                isSelected ? 'ring-2 ring-devtools-accent border-devtools-accent' : ''
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <Icon size={16} className="text-devtools-accent" />
                {info && (
                  <div className="flex items-center space-x-1">
                    {info.trend === 'up' && <TrendingUp size={12} className="text-devtools-success" />}
                    {info.trend === 'down' && <TrendingDown size={12} className="text-devtools-error" />}
                    {info.trend === 'stable' && <Minus size={12} className="text-devtools-textSecondary" />}
                  </div>
                )}
              </div>
              
              <div className="space-y-1">
                <div className="text-lg font-bold text-devtools-text">
                  {latestMetric ? (
                    option.key === 'cache_hit_rate' 
                      ? `${(latestMetric[option.key] * 100).toFixed(1)}%`
                      : (latestMetric[option.key] as number).toFixed(option.key.includes('time') || option.key.includes('overhead') ? 1 : 0)
                  ) : '—'}
                </div>
                
                <div className="text-xs text-devtools-textSecondary">
                  {option.label.split('(')[0].trim()}
                </div>
                
                {info && (
                  <div className="text-xs text-devtools-textSecondary">
                    Avg: {option.key === 'cache_hit_rate' 
                      ? `${(info.avg * 100).toFixed(1)}%`
                      : info.avg.toFixed(option.key.includes('time') || option.key.includes('overhead') ? 1 : 0)}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Chart */}
      <div className="flex-1 p-4">
        {recentMetrics.length > 0 ? (
          <div className="h-full">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-devtools-text">
                {metricOptions.find(o => o.key === selectedMetric)?.label}
              </h3>
              <p className="text-sm text-devtools-textSecondary">
                Real-time performance monitoring over the last {recentMetrics.length} measurements
              </p>
            </div>
            
            <div className="h-full bg-devtools-surface rounded-lg border border-devtools-border p-4">
              <svg
                ref={chartRef}
                width={chartDimensions.width}
                height={chartDimensions.height}
                className="bg-devtools-background rounded"
              />
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 mx-auto bg-devtools-surface rounded-full flex items-center justify-center">
                <Activity size={32} className="text-devtools-textSecondary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-devtools-text mb-2">
                  No Performance Data
                </h3>
                <p className="text-devtools-textSecondary max-w-md">
                  Performance metrics will appear here once debugging sessions are active. 
                  Start processing tokens to see real-time performance charts.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}