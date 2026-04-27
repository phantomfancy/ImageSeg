import styled from 'styled-components'
import { HELP_CONTENT } from '../app/constants'

const Root = styled.div`
  position: fixed;
  inset: 0;
  z-index: 34;
  display: grid;
  place-items: center;

  .help-modal__backdrop {
    position: absolute;
    inset: 0;
    border: none;
    background: var(--modal-backdrop);
    cursor: pointer;
  }

  .help-modal__dialog {
    position: relative;
    z-index: 1;
    width: min(960px, calc(100vw - 40px));
    max-height: calc(100vh - 40px);
    overflow: auto;
    display: grid;
    gap: 18px;
    padding: 22px;
    border: 1px solid var(--shell-border-soft);
    border-radius: 26px;
    background: color-mix(in srgb, var(--paper-strong) 96%, transparent);
    box-shadow: 0 24px 54px rgba(12, 24, 34, 0.22);
    backdrop-filter: blur(18px);
  }

  .help-modal__header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 18px;
  }

  .help-modal__eyebrow {
    display: inline-flex;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 0.76rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .help-modal__title {
    margin: 12px 0 8px;
    font-size: clamp(1.4rem, 2vw, 2rem);
    line-height: 1.1;
  }

  .help-modal__copy {
    margin: 0;
    color: var(--muted);
    line-height: 1.7;
  }

  .help-modal__close {
    min-height: 42px;
    padding: 10px 14px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: color-mix(in srgb, var(--paper) 90%, transparent);
    color: var(--ink);
    cursor: pointer;
  }

  .help-modal__grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }

  .help-modal__card {
    padding: 18px;
    border-radius: 20px;
    border: 1px solid var(--line);
    background: var(--control-bg-soft);
  }

  .help-modal__card h3 {
    margin: 0 0 10px;
    font-size: 1rem;
  }

  .help-modal__card p {
    margin: 0;
    color: var(--muted);
    line-height: 1.7;
  }

  @media (max-width: 1040px) {
    .help-modal__dialog {
      width: min(100vw - 28px, 960px);
      max-height: calc(100vh - 28px);
      padding: 18px;
    }

    .help-modal__grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 720px) {
    .help-modal__header {
      flex-direction: column;
    }

    .help-modal__close {
      width: 100%;
    }
  }
`

type HelpModalProps = {
  onClose: () => void
}

export function HelpModal(props: HelpModalProps) {
  return (
    <Root role="dialog" aria-modal="true" aria-label="帮助说明">
      <button
        type="button"
        className="help-modal__backdrop"
        aria-label="关闭帮助说明"
        onClick={props.onClose}
      />

      <div className="help-modal__dialog">
        <div className="help-modal__header">
          <div>
            <div className="help-modal__eyebrow">帮助</div>
            <h2 className="help-modal__title">工作台使用说明</h2>
            <p className="help-modal__copy">
              这里整理了当前前端工作台最常用的导入、推理、查看与导出说明。
            </p>
          </div>

          <button
            type="button"
            className="help-modal__close"
            onClick={props.onClose}
          >
            关闭
          </button>
        </div>

        <div className="help-modal__grid">
          {HELP_CONTENT.map((item) => (
            <section className="help-modal__card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </section>
          ))}
        </div>
      </div>
    </Root>
  )
}
