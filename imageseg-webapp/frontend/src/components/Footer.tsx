import styled from 'styled-components'

const Root = styled.footer`
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) repeat(3, minmax(0, 1fr));
  gap: 20px;
  margin-top: auto;
  width: 100%;
  padding: 28px 0 8px;
  border-top: 1px solid var(--line);
  color: var(--ink);

  .footer__section {
    display: grid;
    align-content: start;
    gap: 10px;
    min-width: 0;
  }

  .footer__section--brand {
    padding-right: 12px;
  }

  .footer__heading {
    font-size: 0.8rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .footer__title {
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.4;
  }

  .footer__copy {
    margin: 0;
    color: var(--muted);
    line-height: 1.65;
  }

  .footer__meta-list,
  .footer__link-list {
    display: grid;
    gap: 12px;
  }

  .footer__meta-item {
    display: grid;
    gap: 6px;
  }

  .footer__label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .footer__value {
    font-weight: 700;
    line-height: 1.45;
  }

  .footer__link {
    color: var(--sea);
    line-height: 1.6;
    text-decoration: none;
    word-break: break-word;
  }

  .footer__link:hover {
    color: var(--accent);
    text-decoration: underline;
  }

  @media (max-width: 1040px) {
    gap: 16px;
    padding-top: 24px;
  }

  @media (max-width: 720px) {
    gap: 10px;
    font-size: 0.76rem;

    .footer__section {
      gap: 7px;
    }

    .footer__section--brand {
      padding-right: 6px;
    }

    .footer__title {
      font-size: 0.86rem;
    }

    .footer__heading,
    .footer__label {
      font-size: 0.62rem;
      letter-spacing: 0.1em;
    }

    .footer__copy,
    .footer__link,
    .footer__value {
      line-height: 1.45;
    }
  }
`

const PROJECT_REPOSITORY_URL = 'https://github.com/phantomfancy/ImageSeg'
const PROJECT_LICENSE_URL = `${PROJECT_REPOSITORY_URL}/blob/main/LICENSE`
const FOOTER_CONTACT_EMAIL = 'phantomfancy@outlook.com'
const FOOTER_FILING_NUMBER = '备案号待补充'
const FOOTER_CERTIFICATION_INFO = '认证信息待补充'
const COPYRIGHT_TEXT = `Copyright © ${new Date().getFullYear()} phantomfancy`

export function Footer() {
  return (
    <Root>
      <div className="footer__section footer__section--brand">
        <div className="footer__title">ImageSeg-基于onnx的视觉推理工作台</div>
        <p className="footer__copy">{COPYRIGHT_TEXT}</p>
        <p className="footer__copy">项目源码依据 AGPL-3.0 许可进行分发与修改。</p>
      </div>

      <div className="footer__section">
        <div className="footer__heading">联系方式</div>
        <a className="footer__link" href={`mailto:${FOOTER_CONTACT_EMAIL}`}>
          {FOOTER_CONTACT_EMAIL}
        </a>
      </div>

      <div className="footer__section">
        <div className="footer__heading">备案与认证</div>
        <div className="footer__meta-list">
          <div className="footer__meta-item">
            <span className="footer__label">备案信息</span>
            <span className="footer__value">{FOOTER_FILING_NUMBER}</span>
          </div>
          <div className="footer__meta-item">
            <span className="footer__label">认证信息</span>
            <span className="footer__value">{FOOTER_CERTIFICATION_INFO}</span>
          </div>
        </div>
      </div>

      <div className="footer__section">
        <div className="footer__heading">项目链接</div>
        <div className="footer__link-list">
          <a
            className="footer__link"
            href={PROJECT_LICENSE_URL}
            target="_blank"
            rel="noreferrer"
          >
            项目 LICENSE
          </a>
          <a
            className="footer__link"
            href={PROJECT_REPOSITORY_URL}
            target="_blank"
            rel="noreferrer"
          >
            项目仓库
          </a>
        </div>
      </div>
    </Root>
  )
}
