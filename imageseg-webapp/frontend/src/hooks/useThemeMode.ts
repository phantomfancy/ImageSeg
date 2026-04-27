import { useEffect, useState } from 'react'
import type { ResolvedTheme, ThemeMode } from '../app/types'
import {
  getSystemTheme,
  readThemeMode,
  resolveThemeMode,
} from '../app/workspaceUtils'

export function useThemeMode() {
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => readThemeMode())
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>(() => getSystemTheme())
  const resolvedTheme = resolveThemeMode(themeMode, systemTheme)

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const applyThemePreference = (nextMatches: boolean) => {
      setSystemTheme(nextMatches ? 'dark' : 'light')
    }

    applyThemePreference(mediaQuery.matches)

    const handleChange = (event: MediaQueryListEvent) => {
      applyThemePreference(event.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => {
      mediaQuery.removeEventListener('change', handleChange)
    }
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = resolvedTheme
    document.documentElement.style.colorScheme = resolvedTheme
    window.localStorage.setItem('4cimageseg-theme-mode', themeMode)

    return () => {
      delete document.documentElement.dataset.theme
      document.documentElement.style.colorScheme = ''
    }
  }, [resolvedTheme, themeMode])

  return {
    resolvedTheme,
    setThemeMode,
    themeMode,
  }
}
