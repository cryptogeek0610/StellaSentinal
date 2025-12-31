import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'

export function useAnomalies(params: {
  device_id?: number
  start_date?: string
  end_date?: string
  status?: string
  min_score?: number
  max_score?: number
  page?: number
  page_size?: number
}) {
  return useQuery({
    queryKey: ['anomalies', params],
    queryFn: () => api.getAnomalies(params),
  })
}

export function useAnomaly(id: number) {
  return useQuery({
    queryKey: ['anomaly', id],
    queryFn: () => api.getAnomaly(id),
    enabled: !!id,
  })
}

export function useResolveAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, status, notes }: { id: number; status: string; notes?: string }) =>
      api.resolveAnomaly(id, status, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomalies'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })
}

export function useAddNote() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, note, action_type }: { id: number; note: string; action_type?: string }) =>
      api.addNote(id, note, action_type),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', variables.id] })
    },
  })
}

