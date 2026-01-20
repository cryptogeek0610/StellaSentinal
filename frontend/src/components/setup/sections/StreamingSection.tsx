/**
 * Streaming configuration section component.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { ToggleSwitch } from '../../ui';
import { FormField, TextInput, SaveSectionButton } from '../FormComponents';
import { TestConnectionButton } from '../TestConnectionButton';
import type { EnvironmentConfig } from '../../../types/setup';

interface StreamingSectionProps {
  config: EnvironmentConfig;
  updateConfig: <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) => void;
  savingSection: string | null;
  onSave: () => void;
}

export function StreamingSection({
  config,
  updateConfig,
  savingSection,
  onSave
}: StreamingSectionProps) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
        <div>
          <p className="text-sm text-white font-medium">Enable Real-Time Streaming</p>
          <p className="text-xs text-slate-500">Process telemetry data in real-time using Redis pub/sub</p>
        </div>
        <ToggleSwitch
          enabled={config.enable_streaming}
          onChange={(v) => updateConfig('enable_streaming', v)}
          variant="stellar"
          size="lg"
        />
      </div>

      <AnimatePresence>
        {config.enable_streaming && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-6"
          >
            <div className="p-4 rounded-xl bg-orange-500/10 border border-orange-500/30">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-orange-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <div>
                  <p className="text-sm text-orange-400 font-medium">Redis Required</p>
                  <p className="text-xs text-slate-400 mt-1">
                    Real-time streaming requires Redis for pub/sub messaging.
                    The included docker-compose.yml provides a Redis container.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField label="Redis URL" required>
                <TextInput
                  value={config.redis_url}
                  onChange={(v) => updateConfig('redis_url', v)}
                  placeholder="redis://redis:6379"
                />
              </FormField>
              <FormField label="Redis Database">
                <TextInput
                  type="number"
                  value={config.redis_db.toString()}
                  onChange={(v) => updateConfig('redis_db', parseInt(v) || 0)}
                  placeholder="0"
                />
              </FormField>
            </div>

            <div className="p-4 rounded-xl bg-slate-800/30 border border-slate-700/30">
              <h4 className="text-sm font-medium text-white mb-3">Buffer Settings</h4>
              <div className="grid grid-cols-2 gap-4">
                <FormField label="Buffer Size" description="Maximum messages in buffer">
                  <TextInput
                    type="number"
                    value={config.stream_buffer_size.toString()}
                    onChange={(v) => updateConfig('stream_buffer_size', parseInt(v) || 1000)}
                    placeholder="1000"
                  />
                </FormField>
                <FormField label="Flush Interval (ms)" description="How often to flush the buffer">
                  <TextInput
                    type="number"
                    value={config.stream_flush_interval_ms.toString()}
                    onChange={(v) => updateConfig('stream_flush_interval_ms', parseInt(v) || 100)}
                    placeholder="100"
                  />
                </FormField>
              </div>
            </div>

            <TestConnectionButton
              type="redis"
              config={config}
              label="Test Redis Connection"
            />
          </motion.div>
        )}
      </AnimatePresence>
      <SaveSectionButton
        sectionId="streaming"
        isPending={savingSection === 'streaming'}
        onSave={onSave}
      />
    </div>
  );
}
