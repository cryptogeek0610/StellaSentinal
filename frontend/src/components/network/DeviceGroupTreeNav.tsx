/**
 * DeviceGroupTreeNav - Hierarchical navigation for Network Intelligence
 *
 * Features:
 * - Unlimited depth tree with expand/collapse
 * - Device count badges
 * - Visual indication of selected group
 * - SVG icons for groups
 */

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface DeviceGroupNode {
  device_group_id: number;
  group_name: string;
  parent_device_group_id: number | null;
  device_count: number;
  full_path: string;
  children: DeviceGroupNode[];
}

interface DeviceGroupTreeNavProps {
  groups: DeviceGroupNode[];
  selectedGroupId: number | null;
  onSelectGroup: (groupId: number | null) => void;
  isLoading?: boolean;
}

// SVG Icons
function FolderIcon({ expanded, className = 'w-4 h-4' }: { expanded: boolean; className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      {expanded ? (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z"
        />
      ) : (
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
        />
      )}
    </svg>
  );
}

function ChevronIcon({ expanded, className = 'w-4 h-4' }: { expanded: boolean; className?: string }) {
  return (
    <motion.svg
      className={`${className} text-slate-500`}
      animate={{ rotate: expanded ? 90 : 0 }}
      transition={{ duration: 0.15 }}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
    >
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
    </motion.svg>
  );
}

function ServerStackIcon({ className = 'w-4 h-4' }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
      />
    </svg>
  );
}

function TreeNode({
  node,
  depth,
  selectedGroupId,
  onSelect,
  expandedNodes,
  toggleExpand,
}: {
  node: DeviceGroupNode;
  depth: number;
  selectedGroupId: number | null;
  onSelect: (id: number | null) => void;
  expandedNodes: Set<number>;
  toggleExpand: (id: number) => void;
}) {
  const isSelected = selectedGroupId === node.device_group_id;
  const isExpanded = expandedNodes.has(node.device_group_id);
  const hasChildren = node.children.length > 0;

  const baseClasses = 'w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors';
  const hoverClasses = 'hover:bg-slate-700/50';
  const selectedClasses = isSelected
    ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
    : 'text-slate-300';

  return (
    <div>
      <button
        onClick={() => onSelect(node.device_group_id)}
        className={`${baseClasses} ${hoverClasses} ${selectedClasses}`}
        style={{ paddingLeft: `${12 + depth * 16}px` }}
      >
        {hasChildren && (
          <span
            onClick={(e) => {
              e.stopPropagation();
              toggleExpand(node.device_group_id);
            }}
            className="cursor-pointer hover:bg-slate-600/50 rounded p-0.5"
          >
            <ChevronIcon expanded={isExpanded} />
          </span>
        )}
        {!hasChildren && <span className="w-4" />}

        <FolderIcon expanded={isExpanded && hasChildren} />

        <span className="flex-1 text-left truncate">{node.group_name}</span>

        <span className="px-2 py-0.5 rounded-full text-xs bg-slate-700/50 text-slate-400">
          {node.device_count.toLocaleString()}
        </span>
      </button>

      <AnimatePresence>
        {isExpanded && hasChildren && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            {node.children.map((child) => (
              <TreeNode
                key={child.device_group_id}
                node={child}
                depth={depth + 1}
                selectedGroupId={selectedGroupId}
                onSelect={onSelect}
                expandedNodes={expandedNodes}
                toggleExpand={toggleExpand}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function DeviceGroupTreeNav({
  groups,
  selectedGroupId,
  onSelectGroup,
  isLoading,
}: DeviceGroupTreeNavProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<number>>(new Set());

  const toggleExpand = useCallback((id: number) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Auto-expand root nodes on initial render
  const expandAllRoots = useCallback(() => {
    const rootIds = groups.map((g) => g.device_group_id);
    setExpandedNodes(new Set(rootIds));
  }, [groups]);

  // Expand roots when groups change
  useState(() => {
    if (groups.length > 0 && expandedNodes.size === 0) {
      expandAllRoots();
    }
  });

  if (isLoading) {
    return (
      <div className="w-64 border-r border-slate-700/50 bg-slate-800/30 p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-slate-700 rounded w-3/4" />
          <div className="h-4 bg-slate-700 rounded w-1/2 ml-4" />
          <div className="h-4 bg-slate-700 rounded w-2/3 ml-4" />
          <div className="h-4 bg-slate-700 rounded w-1/2 ml-8" />
        </div>
      </div>
    );
  }

  return (
    <div className="w-64 border-r border-slate-700/50 bg-slate-800/30 overflow-y-auto flex-shrink-0">
      <div className="p-4 border-b border-slate-700/50">
        <h3 className="text-sm font-semibold text-white">Device Groups</h3>
        <p className="text-xs text-slate-500 mt-1">Select a group to filter metrics</p>
      </div>

      {/* All Devices option */}
      <div className="p-2">
        <button
          onClick={() => onSelectGroup(null)}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors hover:bg-slate-700/50 ${
            selectedGroupId === null
              ? 'bg-stellar-500/15 text-stellar-400 border border-stellar-500/30'
              : 'text-slate-300'
          }`}
        >
          <ServerStackIcon />
          <span className="flex-1 text-left">All Devices</span>
        </button>
      </div>

      <div className="p-2 space-y-1">
        {groups.map((group) => (
          <TreeNode
            key={group.device_group_id}
            node={group}
            depth={0}
            selectedGroupId={selectedGroupId}
            onSelect={onSelectGroup}
            expandedNodes={expandedNodes}
            toggleExpand={toggleExpand}
          />
        ))}
      </div>
    </div>
  );
}

export default DeviceGroupTreeNav;
