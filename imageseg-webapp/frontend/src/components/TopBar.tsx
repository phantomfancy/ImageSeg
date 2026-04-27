import { useEffect, useRef, useState } from 'react'
import helpIcon from '../assets/help-circle.svg?raw'
import moonIcon from '../assets/moon.svg?raw'
import sunIcon from '../assets/sun.svg?raw'
import styled from 'styled-components'
import type { PageId, ResolvedTheme, ThemeMode } from '../app/types'
import { SvgIcon } from './SvgIcon'

const Root = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px 18px;
  width: 100%;
  padding: 0 0 16px;
  border-bottom: 1px solid var(--line);

  .topbar__brand {
    flex: 0 0 auto;
    min-width: 0;
  }

  .topbar__title {
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.3;
    color: var(--ink);
    white-space: nowrap;
  }

  .topbar__page-nav {
    display: flex;
    flex: 1 1 auto;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    min-width: 0;
  }

  .topbar__nav-item {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 38px;
    padding: 8px 16px;
    border: 1px solid transparent;
    border-radius: 999px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    font-weight: 700;
    white-space: nowrap;
    transition: transform 140ms ease, border-color 140ms ease, background 140ms ease, color 140ms ease;
  }

  .topbar__nav-item:hover {
    transform: translateY(-1px);
    border-color: var(--topbar-action-border);
    background: var(--topbar-action-bg);
    color: var(--ink);
  }

  .topbar__nav-item--active {
    border-color: var(--topbar-action-border);
    background: var(--topbar-action-bg);
    color: var(--accent);
  }

  .topbar__menu-anchor {
    position: relative;
  }

  .topbar__action-list {
    display: flex;
    flex: 0 0 auto;
    align-items: center;
    gap: 10px;
  }

  .topbar__action-button {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    min-height: 38px;
    padding: 8px 12px;
    border: 1px solid var(--topbar-action-border);
    border-radius: 999px;
    background: var(--topbar-action-bg);
    box-shadow: var(--topbar-action-shadow);
    color: var(--ink);
    cursor: pointer;
    transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
  }

  .topbar__action-button:hover {
    transform: translateY(-1px);
    border-color: rgba(199, 88, 43, 0.3);
    background: var(--topbar-action-bg-hover);
  }

  .topbar__menu {
    position: absolute;
    top: calc(100% + 10px);
    right: 0;
    z-index: 20;
    display: grid;
    gap: 6px;
    min-width: 180px;
    padding: 10px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--topbar-menu-bg);
    box-shadow: var(--topbar-menu-shadow);
    backdrop-filter: blur(16px);
  }

  .topbar__menu-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    width: 100%;
    padding: 10px 12px;
    border: none;
    border-radius: 14px;
    background: transparent;
    color: var(--ink);
    cursor: pointer;
    text-align: left;
  }

  .topbar__menu-item:hover {
    background: var(--accent-soft);
  }

  .topbar__menu-item--active {
    background: color-mix(in srgb, var(--accent-soft) 80%, transparent);
    color: var(--accent);
  }

  .topbar__menu-check {
    color: var(--muted);
    font-size: 0.8rem;
  }

  @media (max-width: 1040px) {
    gap: 10px;

    .topbar__title {
      font-size: 0.94rem;
    }

    .topbar__page-nav {
      flex-wrap: nowrap;
      justify-content: center;
      gap: 7px;
    }

    .topbar__nav-item {
      min-height: 34px;
      padding: 7px 13px;
      font-size: 0.86rem;
    }

    .topbar__action-list {
      gap: 8px;
    }

    .topbar__action-button {
      min-height: 34px;
      padding: 7px 10px;
      gap: 8px;
      font-size: 0.86rem;
    }
  }

  @media (max-width: 720px) {
    gap: 8px;

    .topbar__title {
      font-size: 0.84rem;
    }

    .topbar__page-nav {
      flex-direction: row;
      align-items: center;
      gap: 5px;
    }

    .topbar__nav-item {
      min-height: 32px;
      padding: 6px 10px;
      font-size: 0.78rem;
    }

    .topbar__action-list {
      width: auto;
      justify-content: flex-end;
      gap: 6px;
    }

    .topbar__action-button {
      flex: 0 0 auto;
      justify-content: center;
      min-height: 32px;
      padding: 6px 9px;
      gap: 6px;
      font-size: 0.78rem;
    }

    .topbar__menu {
      right: 0;
      min-width: min(220px, calc(100vw - 36px));
    }
  }
`

const PAGE_NAV_ITEMS: ReadonlyArray<{ id: PageId; label: string }> = [
  { id: 'home', label: '首页' },
  { id: 'workspace', label: '工作台' },
]

const THEME_OPTIONS: ReadonlyArray<{ mode: ThemeMode; label: string }> = [
  { mode: 'dark', label: '深色' },
  { mode: 'light', label: '浅色' },
  { mode: 'system', label: '跟随系统' },
]

type TopBarProps = {
  activePageId: PageId
  onHelpOpen: () => void
  onPageNavigate: (pageId: PageId) => void
  onThemeModeSelect: (mode: ThemeMode) => void
  resolvedTheme: ResolvedTheme
  themeMode: ThemeMode
}

export function TopBar(props: TopBarProps) {
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false)
  const themeMenuRef = useRef<HTMLDivElement | null>(null)
  const {
    activePageId,
    onHelpOpen,
    onPageNavigate,
    onThemeModeSelect,
    resolvedTheme,
    themeMode,
  } = props
  const currentThemeIcon = resolvedTheme === 'dark' ? moonIcon : sunIcon
  const currentThemeLabel = themeMode === 'system'
    ? `跟随系统 · ${resolvedTheme === 'dark' ? '深色' : '浅色'}`
    : themeMode === 'dark'
      ? '深色模式'
      : '浅色模式'

  useEffect(() => {
    if (!isThemeMenuOpen) {
      return
    }

    const handlePointerDown = (event: PointerEvent) => {
      if (themeMenuRef.current?.contains(event.target as Node)) {
        return
      }

      setIsThemeMenuOpen(false)
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      setIsThemeMenuOpen(false)
    }

    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isThemeMenuOpen])

  return (
    <Root>
      <div className="topbar__brand">
        <div className="topbar__title">ImageSeg-基于onnx的视觉推理工作台</div>
      </div>

      <nav className="topbar__page-nav" aria-label="页面导航">
        {PAGE_NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`topbar__nav-item${activePageId === item.id ? ' topbar__nav-item--active' : ''}`}
            aria-current={activePageId === item.id ? 'page' : undefined}
            onClick={() => {
              setIsThemeMenuOpen(false)
              onPageNavigate(item.id)
            }}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="topbar__action-list">
        <div className="topbar__menu-anchor" ref={themeMenuRef}>
          <button
            type="button"
            className="topbar__action-button"
            aria-haspopup="menu"
            aria-expanded={isThemeMenuOpen}
            aria-label={`主题切换，当前为${currentThemeLabel}`}
            title={`主题切换，当前为${currentThemeLabel}`}
            onClick={() => {
              setIsThemeMenuOpen((currentValue) => !currentValue)
            }}
          >
            <SvgIcon markup={currentThemeIcon} />
            <span>主题</span>
          </button>

          {isThemeMenuOpen ? (
            <div className="topbar__menu" role="menu" aria-label="主题切换">
              {THEME_OPTIONS.map(({ mode, label }) => (
                <button
                  key={mode}
                  type="button"
                  role="menuitemradio"
                  aria-checked={themeMode === mode}
                  className={`topbar__menu-item${themeMode === mode ? ' topbar__menu-item--active' : ''}`}
                  onClick={() => {
                    onThemeModeSelect(mode)
                    setIsThemeMenuOpen(false)
                  }}
                >
                  <span>{label}</span>
                  {themeMode === mode ? <span className="topbar__menu-check">当前</span> : null}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <button
          type="button"
          className="topbar__action-button"
          aria-label="帮助说明"
          title="帮助说明"
          onClick={() => {
            setIsThemeMenuOpen(false)
            onHelpOpen()
          }}
        >
          <SvgIcon markup={helpIcon} />
          <span>帮助</span>
        </button>
      </div>
    </Root>
  )
}
