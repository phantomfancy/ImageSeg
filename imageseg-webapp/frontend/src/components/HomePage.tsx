import styled from 'styled-components'

const Root = styled.div`
  display: grid;
  gap: 20px;

  .hero {
    min-width: 0;
    padding: 30px 32px;
    border: 1px solid var(--shell-border-soft);
    background:
      linear-gradient(135deg, var(--hero-start), var(--hero-end)),
      linear-gradient(90deg, rgba(248, 245, 237, 0.08), rgba(248, 245, 237, 0.02));
    color: var(--paper);
    border-radius: 28px;
    box-shadow: var(--card-shadow);
    overflow: hidden;
  }

  .hero--home {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
    gap: 28px;
    align-items: stretch;
    min-height: 460px;
  }

  .hero--home::before {
    content: '';
    position: absolute;
    inset: auto -12% -38% 38%;
    height: 58%;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(236, 132, 88, 0.26), transparent 68%);
    pointer-events: none;
  }

  .hero__content {
    position: relative;
    z-index: 1;
    display: grid;
    align-content: center;
  }

  .hero__eyebrow {
    display: inline-flex;
    width: fit-content;
    padding: 6px 12px;
    border-radius: 999px;
    background: var(--contrast-bg-soft);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.75rem;
  }

  .hero h1 {
    margin: 18px 0 12px;
    font-size: clamp(2.2rem, 4vw, 4.2rem);
    line-height: 0.98;
    letter-spacing: -0.05em;
    color: var(--contrast-muted);
  }

  .hero__copy {
    max-width: 70ch;
    margin: 0;
    line-height: 1.7;
    color: var(--hero-copy);
  }

  .hero__actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 24px;
  }

  .hero__action {
    min-height: 46px;
    padding: 12px 18px;
    border-radius: 999px;
    border: 1px solid var(--contrast-border);
    cursor: pointer;
    font-weight: 700;
    transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
  }

  .hero__action:hover {
    transform: translateY(-1px);
  }

  .hero__action--primary {
    border-color: rgba(248, 245, 237, 0.24);
    background: linear-gradient(135deg, var(--accent), #db8d34);
    color: var(--primary-action-text);
  }

  .hero__action--secondary {
    background: var(--contrast-bg-soft);
    color: var(--contrast-muted);
  }

  .hero__signal {
    position: relative;
    z-index: 1;
    display: grid;
    gap: 12px;
    align-content: end;
  }

  .hero__signal-card {
    display: grid;
    gap: 8px;
    padding: 18px;
    border: 1px solid var(--contrast-border);
    border-radius: 20px;
    background: rgba(248, 245, 237, 0.1);
    backdrop-filter: blur(16px);
  }

  .hero__signal-label {
    color: var(--contrast-soft);
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .hero__signal-card strong {
    color: var(--contrast-muted);
    font-size: 1rem;
    line-height: 1.45;
  }

  .home-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
  }

  .home-card {
    padding: 20px;
    border-radius: 24px;
    border: 1px solid var(--line);
    background: color-mix(in srgb, var(--paper-strong) 90%, transparent);
    box-shadow: 0 18px 34px rgba(20, 32, 42, 0.08);
  }

  .home-card__index {
    font-family: var(--mono);
    color: var(--accent);
    font-size: 0.92rem;
  }

  .home-card h2 {
    margin: 14px 0 10px;
    font-size: 1.08rem;
  }

  .home-card p {
    margin: 0;
    color: var(--muted);
    line-height: 1.7;
  }

  @media (max-width: 1040px) {
    .hero {
      padding: 20px;
    }

    .hero--home {
      grid-template-columns: 1fr;
      min-height: 0;
    }

    .hero__signal {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .home-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 720px) {
    .hero__actions {
      gap: 8px;
      margin-top: 18px;
    }

    .hero__action {
      min-height: 40px;
      padding: 10px 14px;
      font-size: 0.86rem;
    }

    .hero__signal {
      grid-template-columns: 1fr;
      gap: 8px;
    }

    .hero__signal-card,
    .home-card {
      padding: 14px;
      border-radius: 16px;
    }
  }
`

type HomePageProps = {
  onEnterWorkspace: () => void
  onHelpOpen: () => void
}

export function HomePage(props: HomePageProps) {
  return (
    <Root>
      <section className="hero hero--home">
        <div className="hero__content">
          <div className="hero__eyebrow">ImageSeg-基于onnx的视觉推理工作台</div>
          <h1>基于onnx的视觉推理工作台</h1>
          <p className="hero__copy">
            面向单图、视频与摄像头场景的本地 ONNX 推理工作台，使用基于CNN或Transformers的目标检测模型进行图像推理。
            导入、推理编排、结果解码和导出全部在前端完成，适合快速验证识别模型。
          </p>
          <div className="hero__actions">
            <button
              type="button"
              className="hero__action hero__action--primary"
              onClick={props.onEnterWorkspace}
            >
              进入工作台
            </button>
            <button
              type="button"
              className="hero__action hero__action--secondary"
              onClick={props.onHelpOpen}
            >
              查看使用说明
            </button>
          </div>
        </div>

        <div className="hero__signal" aria-label="当前能力摘要">
          <div className="hero__signal-card">
            <span className="hero__signal-label">输入源</span>
            <strong>图片 / 视频 / 摄像头</strong>
          </div>
          <div className="hero__signal-card">
            <span className="hero__signal-label">推理后端</span>
            <strong>ONNX Runtime WebGPU</strong>
          </div>
          <div className="hero__signal-card">
            <span className="hero__signal-label">模型契约</span>
            <strong>统一解码与预处理</strong>
          </div>
        </div>
      </section>

      <section className="home-grid" aria-label="首页功能概览">
        <article className="home-card">
          <div className="home-card__index">01</div>
          <h2>纯前端推理</h2>
          <p>输入源、ONNX 会话、结果绘制和导出都在浏览器端完成，减少服务端部署依赖。</p>
        </article>
        <article className="home-card">
          <div className="home-card__index">02</div>
          <h2>统一模型输出</h2>
          <p>通过 Contracts 统一不同检测模型的预处理、输出签名和解码逻辑，降低模型切换成本。</p>
        </article>
        <article className="home-card">
          <div className="home-card__index">03</div>
          <h2>结果复核与导出</h2>
          <p>同一预览区支持输入源和叠加结果切换，图像查看器可缩放、平移并复核框选细节。</p>
        </article>
      </section>
    </Root>
  )
}
