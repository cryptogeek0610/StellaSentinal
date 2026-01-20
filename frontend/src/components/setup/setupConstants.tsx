/**
 * Constants and section definitions for the Setup page.
 */

import type { ReactNode } from 'react';

export interface SetupSection {
  id: string;
  title: string;
  description: string;
  icon: ReactNode;
}

// Section icon components - extracted for reuse
export const SectionIcons = {
  settings: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  database: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
    </svg>
  ),
  server: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
    </svg>
  ),
  mobile: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
    </svg>
  ),
  ai: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
  api: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
    </svg>
  ),
  streaming: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  location: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 11.5a2.5 2.5 0 10-2.5-2.5A2.5 2.5 0 0012 11.5z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.5 9c0 7-7.5 12-7.5 12S4.5 16 4.5 9a7.5 7.5 0 1115 0z" />
    </svg>
  ),
  security: (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
    </svg>
  ),
};

// Section definitions for navigation
export const SECTIONS: SetupSection[] = [
  {
    id: 'environment',
    title: 'Environment',
    description: 'Application settings',
    icon: SectionIcons.settings,
  },
  {
    id: 'backend-db',
    title: 'Backend Database',
    description: 'PostgreSQL configuration',
    icon: SectionIcons.database,
  },
  {
    id: 'xsight-db',
    title: 'XSight Database',
    description: 'SOTI telemetry source',
    icon: SectionIcons.server,
  },
  {
    id: 'mobicontrol-db',
    title: 'MobiControl DB',
    description: 'Device inventory (optional)',
    icon: SectionIcons.mobile,
  },
  {
    id: 'llm',
    title: 'AI / LLM',
    description: 'AI-powered insights',
    icon: SectionIcons.ai,
  },
  {
    id: 'mobicontrol-api',
    title: 'MobiControl API',
    description: 'Real-time data (optional)',
    icon: SectionIcons.api,
  },
  {
    id: 'streaming',
    title: 'Streaming',
    description: 'Real-time processing',
    icon: SectionIcons.streaming,
  },
  {
    id: 'location-sync',
    title: 'Location Sync',
    description: 'Sync location metadata',
    icon: SectionIcons.location,
  },
  {
    id: 'security',
    title: 'Security',
    description: 'API keys & access',
    icon: SectionIcons.security,
  },
];

// LLM provider presets
export const LLM_PRESETS = [
  { name: 'Ollama (Docker)', url: 'http://ollama:11434', model: 'llama3.2', key: 'not-needed' },
  { name: 'Ollama (Local)', url: 'http://host.docker.internal:11434', model: 'llama3.2', key: 'not-needed' },
  { name: 'LM Studio', url: 'http://host.docker.internal:1234', model: 'local-model', key: 'not-needed' },
  { name: 'OpenAI', url: 'https://api.openai.com/v1', model: 'gpt-4-turbo-preview', key: '' },
];
