# UI_AUDIT

## Frontend Stack
- React + Vite entrypoint (`frontend/src/main.tsx:1-20`)
- API client wrapper with fetch + mock mode (`frontend/src/api/client.ts:236-717`)

## Endpoint → UI Mapping
| Endpoint | API client | UI components/pages |
| --- | --- | --- |
| `/dashboard/isolation-forest/stats` | `api.getIsolationForestStats` (`frontend/src/api/client.ts:567-585`) | `IsolationForestViz` (`frontend/src/components/IsolationForestViz.tsx:70-220`), `ModelStatus` page (`frontend/src/pages/ModelStatus.tsx:49-585`), `Dashboard` page (`frontend/src/pages/Dashboard.tsx:96-103`) |
| `/baselines/suggestions` | `api.getBaselineSuggestions` (`frontend/src/api/client.ts:588-607`) | `Baselines` page (`frontend/src/pages/Baselines.tsx:84-706`), `BaselineManagement` (`frontend/src/components/BaselineManagement.tsx:20-278`), `ModelStatus` (`frontend/src/pages/ModelStatus.tsx:57-120`), `Sidebar` (`frontend/src/components/Sidebar.tsx:225-244`) |
| `/baselines/features` | `api.getBaselineFeatures` (`frontend/src/api/client.ts:651-673`) | `Baselines` page (`frontend/src/pages/Baselines.tsx:84-706`) |
| `/baselines/history` | `api.getBaselineHistory` (`frontend/src/api/client.ts:675-694`) | `Baselines` page (`frontend/src/pages/Baselines.tsx:92-706`) |
| `/baselines/analyze-with-llm` | `api.analyzeBaselinesWithLLM` (`frontend/src/api/client.ts:610-628`) | `Baselines` page (`frontend/src/pages/Baselines.tsx:103-706`), `BaselineManagement` (`frontend/src/components/BaselineManagement.tsx:26-278`) |
| `/baselines/apply-adjustment` | `api.applyBaselineAdjustment` (`frontend/src/api/client.ts:631-648`) | `Baselines` page (`frontend/src/pages/Baselines.tsx:113-706`), `BaselineManagement` (`frontend/src/components/BaselineManagement.tsx:47-278`) |
| `/anomalies` | `api.getAnomalies` (`frontend/src/api/client.ts:353-395`) | `Investigations` (`frontend/src/pages/Investigations.tsx:141-352`), `Dashboard` (`frontend/src/pages/Dashboard.tsx:64-73`), `UnifiedDashboard` (`frontend/src/pages/UnifiedDashboard.tsx:76-93`) |
| `/anomalies/grouped` | `api.getGroupedAnomalies` (`frontend/src/api/client.ts:449-482`) | `Investigations` (`frontend/src/pages/Investigations.tsx:162-540`) |

## Issues Found
- Medium: Model stats UI did not distinguish trained config vs defaults, which is now returned by `/dashboard/isolation-forest/stats` (`frontend/src/components/IsolationForestViz.tsx:83-155`).
- Medium: Baseline drift math divided by zero when baseline values were `0`, causing Infinity/NaN in the UI (`frontend/src/pages/Baselines.tsx:403-493`).
- Low: LLM baseline suggestions could hide non-LLM suggestions when LLM returns an empty array (`frontend/src/pages/Baselines.tsx:156-158`, `frontend/src/components/BaselineManagement.tsx:90-93`).
- Medium: No runtime contract validation for critical endpoints allowed silent API drift (`frontend/src/api/client.ts:353-673`).

## Fixes Applied
- Added explicit trained vs default config display for Isolation Forest stats (`frontend/src/components/IsolationForestViz.tsx:83-155`).
- Guarded baseline drift math and visualization widths for zero baselines (`frontend/src/pages/Baselines.tsx:403-493`).
- Fallback to raw suggestions when LLM suggestions are empty (`frontend/src/pages/Baselines.tsx:156-158`, `frontend/src/components/BaselineManagement.tsx:90-93`).
- Added zod-backed contract parsing for isolation-forest stats, baselines, and anomalies (`frontend/src/api/contracts.ts:1-175`, `frontend/src/api/client.ts:353-673`).
- Added frontend contract tests (`frontend/tests/apiContracts.test.ts:1-152`) and wired into verify (`scripts/verify.py:49-68`).

## Manual QA Checklist
1) `/status` (Model Status): open “Model Configuration” and confirm each metric shows trained value with a “Default” line when defaults are present; ensure values match `/dashboard/isolation-forest/stats` response (`frontend/src/components/IsolationForestViz.tsx:83-155`).
2) `/baselines`: with baseline data present, verify drift bars render and do not overflow when baseline is zero; toggle a row to ensure expanded details render (`frontend/src/pages/Baselines.tsx:403-520`).
3) `/baselines`: trigger “Analyze with AI” then verify LLM suggestions appear; if LLM returns empty, raw suggestions still show (`frontend/src/pages/Baselines.tsx:156-158`, `frontend/src/pages/Baselines.tsx:540-706`).
4) `/investigations`: load anomalies list and confirm pagination + status filters still render results; verify grouped view loads without schema errors (`frontend/src/pages/Investigations.tsx:141-540`).
5) `/dashboard/detailed`: verify isolation-forest chart renders without layout shift on load, and no console errors from contract parsing (`frontend/src/components/IsolationForestViz.tsx:159-220`).

