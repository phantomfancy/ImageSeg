import styled from 'styled-components'
import { NAV_ITEMS } from '../app/constants'
import type { SectionId } from '../app/types'

const Root = styled.aside`
  position: fixed;
  top: 50%;
  left: 12px;
  z-index: 8;
  transform: translateY(-50%);
  color: var(--contrast-muted);

  .sidebar__toggle {
    display: inline-grid;
    justify-items: center;
    gap: 3px;
    min-width: 48px;
    min-height: 66px;
    padding: 9px 7px;
    border: 1px solid var(--sidebar-toggle-border);
    border-radius: 999px;
    background: var(--sidebar-toggle-bg);
    box-shadow: var(--sidebar-toggle-shadow);
    color: var(--ink);
    cursor: pointer;
    transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease;
  }

  .sidebar__toggle:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px color-mix(in srgb, var(--ink) 18%, transparent);
  }

  .sidebar__toggle-icon {
    font-size: 1.02rem;
    line-height: 1;
  }

  .sidebar__toggle-text {
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .sidebar__panel {
    position: absolute;
    top: 50%;
    left: 60px;
    width: min(208px, calc(100vw - 124px));
    display: grid;
    gap: 12px;
    padding: 16px 14px;
    border-radius: 22px;
    border: 1px solid var(--sidebar-panel-border);
    background:
      var(--sidebar-panel-bg),
      radial-gradient(circle at top, rgba(199, 88, 43, 0.16), transparent 40%);
    box-shadow: var(--sidebar-panel-shadow);
    opacity: 0;
    pointer-events: none;
    transform: translateY(-50%) translateX(-12px) scale(0.98);
    transition: opacity 160ms ease, transform 160ms ease;
  }

  &.sidebar--expanded .sidebar__panel {
    opacity: 1;
    pointer-events: auto;
    transform: translateY(-50%) translateX(0) scale(1);
  }

  .sidebar__nav {
    display: grid;
    gap: 8px;
  }

  .sidebar__nav-item {
    display: grid;
    align-items: start;
    width: 100%;
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: 16px;
    background: var(--contrast-bg-subtle);
    color: inherit;
    text-align: left;
    cursor: pointer;
    transition: transform 140ms ease, border-color 140ms ease, background 140ms ease;
  }

  .sidebar__nav-item:hover {
    transform: translateY(-1px);
    border-color: var(--contrast-border);
    background: var(--contrast-bg-soft);
  }

  .sidebar__nav-item--active {
    border-color: rgba(199, 88, 43, 0.36);
    background: linear-gradient(135deg, rgba(199, 88, 43, 0.18), var(--contrast-bg-subtle));
  }

  .sidebar__nav-label {
    font-size: 0.94rem;
    font-weight: 700;
    line-height: 1.25;
    color: var(--contrast-muted);
  }

  @media (max-width: 720px) {
    left: 8px;

    .sidebar__toggle {
      min-width: 44px;
      min-height: 60px;
      padding: 8px 6px;
    }

    .sidebar__panel {
      left: 54px;
      width: min(192px, calc(100vw - 96px));
    }

    .sidebar__nav-item {
      padding: 9px 11px;
    }
  }
`

type WorkspaceSidebarProps = {
  activeSectionId: SectionId
  isExpanded: boolean
  onNavigate: (sectionId: SectionId) => void
  onToggle: () => void
}

export function WorkspaceSidebar(props: WorkspaceSidebarProps) {
  return (
    <Root className={props.isExpanded ? 'sidebar--expanded' : undefined}>
      <button
        type="button"
        className="sidebar__toggle"
        aria-expanded={props.isExpanded}
        aria-controls="workspace-navigation"
        aria-label={props.isExpanded ? '收起侧栏导航' : '展开侧栏导航'}
        onClick={props.onToggle}
      >
        <span className="sidebar__toggle-icon" aria-hidden="true">
          {props.isExpanded ? '×' : '≡'}
        </span>
        <span className="sidebar__toggle-text">{props.isExpanded ? '收起' : '导航'}</span>
      </button>

      <div className="sidebar__panel" aria-hidden={!props.isExpanded}>
        <nav
          id="workspace-navigation"
          className="sidebar__nav"
          aria-label="工作台区块导航"
        >
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebar__nav-item${props.activeSectionId === item.id ? ' sidebar__nav-item--active' : ''}`}
              aria-current={props.activeSectionId === item.id ? 'location' : undefined}
              onClick={() => {
                props.onNavigate(item.id)
              }}
            >
              <span className="sidebar__nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
      </div>
    </Root>
  )
}
