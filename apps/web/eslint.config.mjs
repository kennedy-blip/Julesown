import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTypeScript from 'eslint-config-next/typescript';

export default defineConfig([
  ...nextVitals,
  ...nextTypeScript,
  {
    rules: {
      // The dashboard runs in the browser only: it reads what the browser has
      // stored and loads from the API after the page is on screen, in effects
      // that then set state. That is what this rule reports, every time.
      'react-hooks/set-state-in-effect': 'off',
    },
  },
  globalIgnores(['.next/**', 'next-env.d.ts', 'test-results/**', 'playwright-report/**']),
]);
