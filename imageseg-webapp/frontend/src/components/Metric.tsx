import styled from 'styled-components'

const Root = styled.div`
  padding: 14px;
  border-radius: 18px;
  background: var(--metric-bg);
  border: 1px solid var(--line);
`

const Label = styled.div`
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
`

const Value = styled.div`
  margin-top: 8px;
  font-weight: 700;
  line-height: 1.4;
  word-break: break-word;
`

type MetricProps = {
  label: string
  value: string
}

export function Metric(props: MetricProps) {
  return (
    <Root>
      <Label>{props.label}</Label>
      <Value>{props.value}</Value>
    </Root>
  )
}
