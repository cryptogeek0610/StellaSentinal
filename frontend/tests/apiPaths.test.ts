import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from '../src/api/client';

describe('api paths', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    mockFetch.mockReset();
  });

  it('calls cost alerts endpoint', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ alerts: [], total: 0 }),
    });

    await api.getCostAlerts();
    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/costs/alerts');
  });

  it('calls event alert summary endpoint', async () => {
    await api.getAlertSummary();
    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/insights/events/alerts/summary');
  });

  it('calls correlations endpoint', async () => {
    await api.getCorrelationMatrix();
    const [url] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/correlations/matrix');
  });
});
