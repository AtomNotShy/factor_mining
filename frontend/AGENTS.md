# AGENTS.md - Frontend Development Guide

**Generated:** 2025-01-10 | **Commit:** Current working tree

---

## OVERVIEW

React/TypeScript frontend with Vite for factor mining system backtest interface.

---

## STRUCTURE

```
frontend/
├── src/
│   ├── components/  # Layout, BacktestForm/Results, charts (Recharts, TradingView)
│   ├── pages/       # Dashboard, Backtest, History, Monitoring, Settings
│   ├── services/    # Axios API (proxied /api → localhost:8000)
│   ├── stores/      # Zustand (theme)
│   ├── i18n/        # i18next (zh/en)
│   ├── utils/       # Helpers
│   └── main.tsx
├── dist/
└── vite.config.ts   # Port 3000
```
---

## WHERE TO LOOK

Start: `pnpm dev` (port 3000, /api → localhost:8000) | Add page: `src/pages/` → `App.tsx` routes → `Layout.tsx` nav | API: `src/services/api.ts` (Axios) | State: `src/stores/themeStore.ts` (Zustand) | Charts: `src/components/charts/` (Recharts + ECharts) | i18n: `src/i18n/config.tsx`

---

## CONVENTIONS

Components: Functional with hooks, interface props. Imports: React → third-party → local, `@/*` alias → `./src/*`. State: Zustand (global), useState (local). Styling: Tailwind CSS, dark mode `'class'` strategy. TypeScript: Strict mode enabled. Design: Stripe/Vercel/Linear/GOV.UK (typography-first, minimal icons, mobile-first).

---

## ANTI-PATTERNS

**CRITICAL:** No test structure (no test files/framework), no error boundaries (React errors crash entire app), no loading states (API calls hang without feedback).

**Code Quality:** Mixed charting (both Recharts and ECharts), no ESLint config (script exists, no .eslintrc), missing env validation (no .env file, no VITE_ prefix checks).

**Performance:** No code splitting (single bundle), no memoization (expensive chart recalculations), no debouncing (search/filter rapid re-renders).
