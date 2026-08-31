import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

export default defineConfig(({ mode }) => {
  const single = mode !== 'multi'

  return {
    base: './',
    plugins: [
      react(),
      tailwindcss(),
      ...(single
        ? [viteSingleFile({ removeViteModuleLoader: true, useRecommendedBuildConfig: true })]
        : []),
    ],
    build: {
      target: 'es2020',
      outDir: 'dist',
      emptyOutDir: true,
      sourcemap: false,
      cssCodeSplit: false,
      assetsInlineLimit: 100_000_000,
      reportCompressedSize: false,
      chunkSizeWarningLimit: 20_000,
      modulePreload: { polyfill: false },
      rollupOptions: {
        output: { inlineDynamicImports: single, manualChunks: undefined },
      },
    },
    server: { port: 5173, host: true },
    preview: { port: 8080, host: true },
  }
})
