import styled from 'styled-components'

const Root = styled.span`
  display: inline-grid;
  place-items: center;
  width: 1.1rem;
  height: 1.1rem;
  color: currentColor;

  svg {
    width: 100%;
    height: 100%;
    display: block;
  }
`

type SvgIconProps = {
  markup: string
}

export function SvgIcon(props: SvgIconProps) {
  return <Root aria-hidden="true" dangerouslySetInnerHTML={{ __html: props.markup }} />
}
