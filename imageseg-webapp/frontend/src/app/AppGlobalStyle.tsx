import { createGlobalStyle } from 'styled-components'

export const AppGlobalStyle = createGlobalStyle`
  :root {
    --paper: #f8f5ed;
    --paper-strong: rgba(248, 245, 237, 0.94);
    --ink: #14202a;
    --muted: #56636d;
    --line: rgba(20, 32, 42, 0.12);
    --accent: #c7582b;
    --accent-soft: rgba(199, 88, 43, 0.12);
    --sea: #0e5a74;
    --card-shadow: 0 28px 60px rgba(20, 32, 42, 0.12);
    --page-glow-warm: rgba(199, 88, 43, 0.18);
    --page-glow-cool: rgba(14, 90, 116, 0.18);
    --page-bg-top: #d8e5ea;
    --page-bg-mid: #efe7d7;
    --page-bg-bottom: #f3eee4;
    --shell-border-soft: rgba(248, 245, 237, 0.4);
    --hero-start: #f6efe2;
    --hero-end: #dce9ee;
    --hero-copy: rgba(20, 32, 42, 0.78);
    --hero-title: #14202a;
    --hero-badge-bg: rgba(20, 32, 42, 0.06);
    --hero-badge-text: rgba(20, 32, 42, 0.72);
    --hero-primary-action-text: #14202a;
    --hero-signal-label: rgba(20, 32, 42, 0.58);
    --hero-signal-text: #14202a;
    --hero-secondary-action-bg: rgba(20, 32, 42, 0.06);
    --hero-secondary-action-text: #14202a;
    --hero-secondary-action-border: rgba(20, 32, 42, 0.1);
    --hero-glow: radial-gradient(circle, rgba(199, 88, 43, 0.18), transparent 68%);
    --home-signal-card-bg: linear-gradient(180deg, rgba(255, 252, 247, 0.86), rgba(244, 248, 250, 0.78));
    --home-signal-card-border: rgba(20, 32, 42, 0.08);
    --home-signal-card-shadow: 0 16px 28px rgba(20, 32, 42, 0.08);
    --home-card-bg: linear-gradient(180deg, rgba(255, 252, 247, 0.98), rgba(244, 248, 250, 0.92));
    --home-card-border: rgba(20, 32, 42, 0.08);
    --home-card-shadow: 0 18px 34px rgba(20, 32, 42, 0.07);
    --home-card-title: #14202a;
    --home-card-body: #56636d;
    --control-bg: rgba(255, 255, 255, 0.72);
    --control-bg-soft: rgba(255, 255, 255, 0.58);
    --soft-fill: rgba(14, 90, 116, 0.05);
    --preview-stage-bg: linear-gradient(180deg, rgba(14, 90, 116, 0.08), rgba(20, 32, 42, 0.06));
    --sidebar-toggle-bg: rgba(248, 245, 237, 0.92);
    --sidebar-toggle-border: rgba(20, 32, 42, 0.14);
    --sidebar-toggle-shadow: 0 10px 22px rgba(20, 32, 42, 0.12);
    --sidebar-panel-border: rgba(20, 32, 42, 0.1);
    --sidebar-panel-bg: linear-gradient(180deg, rgba(248, 245, 237, 0.96), rgba(231, 240, 244, 0.92));
    --sidebar-panel-glow: radial-gradient(circle at top, rgba(199, 88, 43, 0.16), transparent 42%);
    --sidebar-panel-shadow: 0 16px 34px rgba(20, 32, 42, 0.12);
    --sidebar-nav-item-bg: rgba(20, 32, 42, 0.04);
    --sidebar-nav-item-bg-hover: rgba(20, 32, 42, 0.08);
    --sidebar-nav-item-bg-active: linear-gradient(135deg, rgba(199, 88, 43, 0.18), rgba(20, 32, 42, 0.04));
    --sidebar-nav-item-border-hover: rgba(20, 32, 42, 0.12);
    --sidebar-nav-item-border-active: rgba(199, 88, 43, 0.36);
    --sidebar-nav-text: rgba(20, 32, 42, 0.86);
    --topbar-action-bg: rgba(248, 245, 237, 0.72);
    --topbar-action-bg-hover: rgba(248, 245, 237, 0.92);
    --topbar-action-border: rgba(20, 32, 42, 0.14);
    --topbar-action-shadow: none;
    --topbar-menu-bg: rgba(248, 245, 237, 0.96);
    --topbar-menu-shadow: 0 18px 34px rgba(20, 32, 42, 0.14);
    --modal-backdrop: rgba(20, 32, 42, 0.7);
    --modal-surface: linear-gradient(180deg, rgba(20, 32, 42, 0.96), rgba(15, 31, 43, 0.94));
    --modal-viewport-bg:
      linear-gradient(180deg, rgba(248, 245, 237, 0.06), rgba(248, 245, 237, 0.02)),
      rgba(248, 245, 237, 0.02);
    --modal-video-bg: #060b0f;
    --modal-video-outline: rgba(248, 245, 237, 0.08);
    --preview-zoom-surface: linear-gradient(180deg, rgba(248, 245, 237, 0.98), rgba(239, 231, 215, 0.96));
    --preview-zoom-viewport-bg:
      linear-gradient(180deg, rgba(14, 90, 116, 0.06), rgba(20, 32, 42, 0.04)),
      rgba(20, 32, 42, 0.03);
    --preview-zoom-text: #14202a;
    --preview-zoom-text-muted: rgba(20, 32, 42, 0.78);
    --preview-zoom-text-soft: rgba(20, 32, 42, 0.66);
    --preview-zoom-control-bg: rgba(20, 32, 42, 0.06);
    --preview-zoom-control-bg-hover: rgba(20, 32, 42, 0.1);
    --preview-zoom-control-border: rgba(20, 32, 42, 0.14);
    --preview-zoom-control-active-bg: rgba(199, 88, 43, 0.14);
    --preview-zoom-control-active-border: rgba(199, 88, 43, 0.34);
    --preview-zoom-close-bg: rgba(199, 88, 43, 0.12);
    --contrast-soft: rgba(248, 245, 237, 0.76);
    --contrast-muted: rgba(248, 245, 237, 0.82);
    --contrast-bg-soft: rgba(248, 245, 237, 0.08);
    --contrast-bg-subtle: rgba(248, 245, 237, 0.06);
    --contrast-border: rgba(248, 245, 237, 0.12);
    --contrast-border-strong: rgba(248, 245, 237, 0.16);
    --floating-button-border: rgba(248, 245, 237, 0.22);
    --floating-button-bg: rgba(20, 32, 42, 0.5);
    --floating-button-bg-hover: rgba(20, 32, 42, 0.62);
    --floating-button-text: #fff8f2;
    --primary-action-text: #fff8f2;
    --secondary-action-border: rgba(14, 90, 116, 0.18);
    --secondary-action-bg: rgba(14, 90, 116, 0.08);
    --mode-switch-border: rgba(20, 32, 42, 0.12);
    --mode-switch-bg: rgba(255, 255, 255, 0.64);
    --mode-switch-active-border: rgba(199, 88, 43, 0.48);
    --mode-switch-active-bg: rgba(199, 88, 43, 0.12);
    --metric-bg: linear-gradient(180deg, rgba(255, 255, 255, 0.76), rgba(255, 255, 255, 0.54));
    --download-link-bg: rgba(199, 88, 43, 0.1);
    --download-link-border: rgba(199, 88, 43, 0.22);
    --download-link-hover-bg: rgba(199, 88, 43, 0.14);
    --sans: 'Segoe UI Variable', 'Bahnschrift', 'Microsoft YaHei UI', sans-serif;
    --mono: 'Cascadia Code', 'Consolas', monospace;
  }

  [data-theme='dark'] {
    --paper: #13202b;
    --paper-strong: rgba(19, 32, 43, 0.94);
    --ink: #eef4f7;
    --muted: #9fb0bc;
    --line: rgba(238, 244, 247, 0.12);
    --accent: #ec8458;
    --accent-soft: rgba(236, 132, 88, 0.16);
    --sea: #67bfd8;
    --card-shadow: 0 28px 60px rgba(2, 8, 14, 0.34);
    --page-glow-warm: rgba(236, 132, 88, 0.12);
    --page-glow-cool: rgba(38, 141, 178, 0.16);
    --page-bg-top: #071018;
    --page-bg-mid: #0b1822;
    --page-bg-bottom: #111f2a;
    --shell-border-soft: rgba(238, 244, 247, 0.12);
    --hero-start: rgba(18, 74, 94, 0.86);
    --hero-end: rgba(7, 16, 24, 0.96);
    --hero-copy: rgba(238, 244, 247, 0.86);
    --hero-title: rgba(238, 244, 247, 0.82);
    --hero-badge-bg: rgba(238, 244, 247, 0.08);
    --hero-badge-text: rgba(238, 244, 247, 0.78);
    --hero-primary-action-text: #fff8f2;
    --hero-signal-label: rgba(238, 244, 247, 0.68);
    --hero-signal-text: rgba(238, 244, 247, 0.84);
    --hero-secondary-action-bg: rgba(238, 244, 247, 0.08);
    --hero-secondary-action-text: rgba(238, 244, 247, 0.82);
    --hero-secondary-action-border: rgba(238, 244, 247, 0.12);
    --hero-glow: radial-gradient(circle, rgba(236, 132, 88, 0.26), transparent 68%);
    --home-signal-card-bg: rgba(238, 244, 247, 0.08);
    --home-signal-card-border: rgba(238, 244, 247, 0.12);
    --home-signal-card-shadow: 0 18px 32px rgba(2, 8, 14, 0.22);
    --home-card-bg: linear-gradient(180deg, rgba(17, 29, 40, 0.9), rgba(12, 21, 30, 0.84));
    --home-card-border: rgba(238, 244, 247, 0.1);
    --home-card-shadow: 0 18px 34px rgba(2, 8, 14, 0.22);
    --home-card-title: #eef4f7;
    --home-card-body: #9fb0bc;
    --control-bg: rgba(15, 27, 37, 0.88);
    --control-bg-soft: rgba(15, 27, 37, 0.74);
    --soft-fill: rgba(103, 191, 216, 0.08);
    --preview-stage-bg: linear-gradient(180deg, rgba(103, 191, 216, 0.08), rgba(238, 244, 247, 0.04));
    --sidebar-toggle-bg: rgba(19, 32, 43, 0.94);
    --sidebar-toggle-border: rgba(238, 244, 247, 0.12);
    --sidebar-toggle-shadow: 0 10px 22px rgba(2, 8, 14, 0.28);
    --sidebar-panel-border: rgba(238, 244, 247, 0.12);
    --sidebar-panel-bg: linear-gradient(180deg, rgba(8, 17, 24, 0.96), rgba(16, 33, 46, 0.94));
    --sidebar-panel-glow: radial-gradient(circle at top, rgba(236, 132, 88, 0.16), transparent 40%);
    --sidebar-panel-shadow: 0 18px 36px rgba(2, 8, 14, 0.3);
    --sidebar-nav-item-bg: rgba(238, 244, 247, 0.06);
    --sidebar-nav-item-bg-hover: rgba(238, 244, 247, 0.08);
    --sidebar-nav-item-bg-active: linear-gradient(135deg, rgba(236, 132, 88, 0.2), rgba(238, 244, 247, 0.04));
    --sidebar-nav-item-border-hover: rgba(238, 244, 247, 0.12);
    --sidebar-nav-item-border-active: rgba(236, 132, 88, 0.38);
    --sidebar-nav-text: rgba(238, 244, 247, 0.84);
    --topbar-action-bg: rgba(17, 29, 40, 0.86);
    --topbar-action-bg-hover: rgba(22, 38, 52, 0.96);
    --topbar-action-border: rgba(238, 244, 247, 0.12);
    --topbar-action-shadow: none;
    --topbar-menu-bg: rgba(13, 24, 34, 0.98);
    --topbar-menu-shadow: 0 18px 34px rgba(2, 8, 14, 0.28);
    --modal-backdrop: rgba(2, 8, 14, 0.82);
    --modal-surface: linear-gradient(180deg, rgba(9, 17, 24, 0.98), rgba(13, 25, 35, 0.96));
    --modal-viewport-bg:
      linear-gradient(180deg, rgba(238, 244, 247, 0.06), rgba(238, 244, 247, 0.02)),
      rgba(8, 17, 24, 0.56);
    --modal-video-bg: #02070c;
    --modal-video-outline: rgba(238, 244, 247, 0.08);
    --preview-zoom-surface: linear-gradient(180deg, rgba(9, 17, 24, 0.98), rgba(13, 25, 35, 0.96));
    --preview-zoom-viewport-bg:
      linear-gradient(180deg, rgba(238, 244, 247, 0.06), rgba(238, 244, 247, 0.02)),
      rgba(8, 17, 24, 0.56);
    --preview-zoom-text: #eef4f7;
    --preview-zoom-text-muted: rgba(238, 244, 247, 0.82);
    --preview-zoom-text-soft: rgba(238, 244, 247, 0.78);
    --preview-zoom-control-bg: rgba(238, 244, 247, 0.08);
    --preview-zoom-control-bg-hover: rgba(238, 244, 247, 0.12);
    --preview-zoom-control-border: rgba(238, 244, 247, 0.12);
    --preview-zoom-control-active-bg: rgba(236, 132, 88, 0.2);
    --preview-zoom-control-active-border: rgba(236, 132, 88, 0.36);
    --preview-zoom-close-bg: rgba(236, 132, 88, 0.16);
    --contrast-soft: rgba(238, 244, 247, 0.78);
    --contrast-muted: rgba(238, 244, 247, 0.82);
    --contrast-bg-soft: rgba(238, 244, 247, 0.08);
    --contrast-bg-subtle: rgba(238, 244, 247, 0.06);
    --contrast-border: rgba(238, 244, 247, 0.12);
    --contrast-border-strong: rgba(238, 244, 247, 0.14);
    --floating-button-border: rgba(238, 244, 247, 0.18);
    --floating-button-bg: rgba(7, 16, 24, 0.58);
    --floating-button-bg-hover: rgba(12, 23, 33, 0.74);
    --floating-button-text: #eef4f7;
    --primary-action-text: #081119;
    --secondary-action-border: rgba(103, 191, 216, 0.28);
    --secondary-action-bg: rgba(103, 191, 216, 0.12);
    --mode-switch-border: rgba(238, 244, 247, 0.14);
    --mode-switch-bg: rgba(15, 27, 37, 0.82);
    --mode-switch-active-border: rgba(236, 132, 88, 0.52);
    --mode-switch-active-bg: rgba(236, 132, 88, 0.18);
    --metric-bg: linear-gradient(180deg, rgba(17, 29, 40, 0.88), rgba(12, 21, 30, 0.8));
    --download-link-bg: rgba(236, 132, 88, 0.18);
    --download-link-border: rgba(236, 132, 88, 0.28);
    --download-link-hover-bg: rgba(236, 132, 88, 0.24);
  }

  html {
    scroll-behavior: smooth;
  }

  * {
    box-sizing: border-box;
  }

  body,
  #root {
    min-height: 100vh;
  }

  body {
    margin: 0;
    min-width: 320px;
    color: var(--ink);
    background:
      radial-gradient(circle at top left, var(--page-glow-warm), transparent 28%),
      radial-gradient(circle at top right, var(--page-glow-cool), transparent 34%),
      linear-gradient(180deg, var(--page-bg-top) 0%, var(--page-bg-mid) 46%, var(--page-bg-bottom) 100%);
    font-family: var(--sans);
  }

  button,
  input,
  select,
  a {
    font: inherit;
  }

  .app-shell {
    --section-scroll-offset: 28px;
    min-height: 100vh;
    padding: 20px 28px;
    position: relative;
  }

  .page-layout {
    display: flex;
    min-height: calc(100vh - 40px);
    flex-direction: column;
    gap: 32px;
  }

  .surface {
    display: flex;
    flex-direction: column;
    gap: 20px;
    min-width: 0;
  }

  .shell {
    width: min(1360px, 100%);
    margin: 0 auto;
    padding: 16px 0 0;
  }

  [data-nav-section] {
    scroll-margin-top: var(--section-scroll-offset);
  }

  @media (max-width: 1040px) {
    .app-shell {
      padding: 16px 22px;
    }

    .page-layout {
      gap: 28px;
    }

    .shell {
      padding-top: 0;
    }
  }

  @media (max-width: 720px) {
    .app-shell {
      padding: 12px 18px;
    }

    .page-layout {
      gap: 24px;
    }
  }
`
