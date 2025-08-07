'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { RoutingEvent, ExpertMetrics, VisualizationConfig, FilterConfig } from '@/types';

interface Props {
  routingEvents: RoutingEvent[];
  expertMetrics: ExpertMetrics[];
  config: VisualizationConfig;
  filter: FilterConfig;
  isPaused: boolean;
}

interface VisualizationNode {
  id: string;
  type: 'token' | 'expert';
  label: string;
  x: number;
  y: number;
  tokenIdx?: number;
  expertId?: number;
  layerIdx?: number;
  confidence?: number;
  weight?: number;
  color?: string;
}

interface VisualizationLink {
  source: VisualizationNode;
  target: VisualizationNode;
  weight: number;
  confidence: number;
}

const EXPERT_COLORS = [
  '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',
  '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd'
];

export function RoutingVisualization({ 
  routingEvents, 
  expertMetrics, 
  config, 
  filter, 
  isPaused 
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState<VisualizationNode | null>(null);
  
  // Resize observer
  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });

    const container = svgRef.current?.parentElement;
    if (container) {
      resizeObserver.observe(container);
    }

    return () => resizeObserver.disconnect();
  }, []);

  // Process routing events into visualization data
  const processVisualizationData = (events: RoutingEvent[]) => {
    const nodes: VisualizationNode[] = [];
    const links: VisualizationLink[] = [];
    
    // Filter events based on current filter config
    const filteredEvents = events.filter(event => {
      return event.layer_idx >= filter.layerRange[0] && 
             event.layer_idx <= filter.layerRange[1] &&
             event.routing_weights.some((weight, idx) => 
               weight >= filter.confidenceThreshold && 
               idx >= filter.expertRange[0] && 
               idx <= filter.expertRange[1]
             );
    });

    const displayEvents = filteredEvents.slice(-config.maxTokensDisplay);
    
    // Create token nodes
    displayEvents.forEach((event, eventIdx) => {
      const tokenNode: VisualizationNode = {
        id: `token-${event.id}`,
        type: 'token',
        label: event.token_text,
        x: 100,
        y: 100 + (eventIdx * 40),
        tokenIdx: event.token_idx,
        layerIdx: event.layer_idx,
        color: '#cccccc'
      };
      nodes.push(tokenNode);

      // Create expert nodes and links
      event.selected_experts.forEach((expertId, expertIdx) => {
        const confidence = event.confidence_scores[expertIdx] || 0;
        const weight = event.routing_weights[expertId] || 0;
        
        if (weight < filter.confidenceThreshold) return;

        const expertNodeId = `expert-${event.layer_idx}-${expertId}`;
        let expertNode = nodes.find(n => n.id === expertNodeId);
        
        if (!expertNode) {
          expertNode = {
            id: expertNodeId,
            type: 'expert',
            label: `E${expertId}`,
            x: 400 + (expertId * 80),
            y: 200 + (event.layer_idx * 60),
            expertId,
            layerIdx: event.layer_idx,
            color: EXPERT_COLORS[expertId % EXPERT_COLORS.length],
            weight: 0
          };
          nodes.push(expertNode);
        }

        // Update expert weight (for sizing)
        expertNode.weight = (expertNode.weight || 0) + weight;

        // Create link
        links.push({
          source: tokenNode,
          target: expertNode,
          weight,
          confidence
        });
      });
    });

    return { nodes, links };
  };

  // Main visualization effect
  useEffect(() => {
    if (!svgRef.current || routingEvents.length === 0 || isPaused) {
      return;
    }

    const svg = d3.select(svgRef.current);
    const { width, height } = dimensions;
    
    // Clear previous content
    svg.selectAll('*').remove();

    // Process data
    const { nodes, links } = processVisualizationData(routingEvents);
    
    if (nodes.length === 0) return;

    // Create scales
    const linkOpacityScale = d3.scaleLinear()
      .domain(d3.extent(links, d => d.confidence) as [number, number])
      .range([0.2, 0.8]);

    const linkWidthScale = d3.scaleLinear()
      .domain(d3.extent(links, d => d.weight) as [number, number])
      .range([1, 5]);

    const expertSizeScale = d3.scaleLinear()
      .domain(d3.extent(nodes.filter(n => n.type === 'expert'), d => d.weight || 0) as [number, number])
      .range([20, 60]);

    // Create main group with zoom behavior
    const g = svg.append('g');
    
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });
    
    svg.call(zoom);

    // Create layout - arrange nodes in layers
    const tokenNodes = nodes.filter(n => n.type === 'token');
    const expertNodes = nodes.filter(n => n.type === 'expert');
    
    // Position token nodes on the left
    tokenNodes.forEach((node, i) => {
      node.x = 100;
      node.y = height / 2 - (tokenNodes.length * 20) + (i * 40);
    });
    
    // Position expert nodes by layer and expert ID
    expertNodes.forEach(node => {
      node.x = 300 + (node.layerIdx || 0) * 200;
      node.y = 100 + (node.expertId || 0) * 60;
    });

    // Draw links
    const linkGroup = g.append('g').attr('class', 'links');
    
    const linkElements = linkGroup
      .selectAll('.routing-line')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'routing-line')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', d => d.target.color || '#666')
      .attr('stroke-width', d => linkWidthScale(d.weight))
      .attr('stroke-opacity', d => linkOpacityScale(d.confidence))
      .style('stroke-dasharray', '5,5');

    // Animate links if enabled
    if (config.animateRouting) {
      linkElements
        .style('stroke-dashoffset', 0)
        .transition()
        .duration(2000)
        .ease(d3.easeLinear)
        .style('stroke-dashoffset', 20)
        .on('end', function() {
          d3.select(this).style('stroke-dashoffset', 0);
        });
    }

    // Draw nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup
      .selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node expert-node')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .style('cursor', 'pointer')
      .on('mouseover', (event, d) => {
        setHoveredNode(d);
        
        // Highlight connected links
        linkElements
          .attr('stroke-opacity', link => 
            link.source.id === d.id || link.target.id === d.id 
              ? linkOpacityScale(link.confidence) * 2 
              : 0.1
          );
      })
      .on('mouseout', () => {
        setHoveredNode(null);
        linkElements.attr('stroke-opacity', d => linkOpacityScale(d.confidence));
      });

    // Draw node circles
    nodeElements
      .append('circle')
      .attr('r', d => {
        if (d.type === 'expert') {
          return expertSizeScale(d.weight || 0) / 2;
        }
        return 15; // Token nodes
      })
      .attr('fill', d => d.color || '#666')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    // Draw node labels
    if (config.showTokenLabels) {
      nodeElements
        .append('text')
        .attr('dy', d => d.type === 'expert' ? 5 : -20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#cccccc')
        .attr('font-size', '12px')
        .attr('font-family', 'Monaco, monospace')
        .text(d => d.label);
    }

    // Draw confidence scores if enabled
    if (config.showConfidenceScores && expertNodes.length > 0) {
      expertNodes.forEach(expertNode => {
        const connectedLinks = links.filter(l => l.target.id === expertNode.id);
        const avgConfidence = connectedLinks.length > 0 
          ? connectedLinks.reduce((sum, l) => sum + l.confidence, 0) / connectedLinks.length
          : 0;
        
        if (avgConfidence > 0) {
          nodeElements
            .filter(d => d.id === expertNode.id)
            .append('text')
            .attr('dy', 25)
            .attr('text-anchor', 'middle')
            .attr('fill', '#999')
            .attr('font-size', '10px')
            .text(`${(avgConfidence * 100).toFixed(1)}%`);
        }
      });
    }

    // Add layer labels
    const layers = [...new Set(nodes.map(n => n.layerIdx).filter(l => l !== undefined))];
    layers.forEach(layerIdx => {
      g.append('text')
        .attr('x', 300 + layerIdx! * 200)
        .attr('y', 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#666')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(`Layer ${layerIdx}`);
    });

    // Auto-fit view to content
    const bounds = g.node()?.getBBox();
    if (bounds) {
      const fullWidth = bounds.width;
      const fullHeight = bounds.height;
      const midX = bounds.x + fullWidth / 2;
      const midY = bounds.y + fullHeight / 2;
      
      const scale = 0.9 / Math.max(fullWidth / width, fullHeight / height);
      const translate = [width / 2 - scale * midX, height / 2 - scale * midY];
      
      svg.transition()
        .duration(750)
        .call(
          zoom.transform,
          d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
        );
    }

  }, [routingEvents, config, filter, dimensions, isPaused]);

  return (
    <div className="relative w-full h-full">
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        className="bg-devtools-background"
      />
      
      {/* Tooltip */}
      {hoveredNode && (
        <div 
          className="absolute bg-devtools-surface border border-devtools-border rounded-lg p-3 pointer-events-none z-10 shadow-lg"
          style={{
            left: hoveredNode.x + 20,
            top: hoveredNode.y - 10,
          }}
        >
          <div className="text-sm">
            <div className="font-semibold text-devtools-text">
              {hoveredNode.type === 'expert' 
                ? `Expert ${hoveredNode.expertId}` 
                : `Token: ${hoveredNode.label}`
              }
            </div>
            {hoveredNode.layerIdx !== undefined && (
              <div className="text-devtools-textSecondary">
                Layer {hoveredNode.layerIdx}
              </div>
            )}
            {hoveredNode.weight !== undefined && (
              <div className="text-devtools-textSecondary">
                Weight: {hoveredNode.weight.toFixed(3)}
              </div>
            )}
            {hoveredNode.confidence !== undefined && (
              <div className="text-devtools-textSecondary">
                Confidence: {(hoveredNode.confidence * 100).toFixed(1)}%
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="absolute top-4 right-4 bg-devtools-surface border border-devtools-border rounded-lg p-3">
        <div className="text-sm space-y-2">
          <div className="font-semibold text-devtools-text">Legend</div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-gray-500"></div>
            <span className="text-devtools-textSecondary text-xs">Tokens</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span className="text-devtools-textSecondary text-xs">Experts</span>
          </div>
          <div className="text-xs text-devtools-textSecondary">
            Line width = routing weight<br/>
            Line opacity = confidence
          </div>
        </div>
      </div>
    </div>
  );
}