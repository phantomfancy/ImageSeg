import styled from 'styled-components'

const Root = styled.div`
  margin-top: 18px;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: var(--control-bg-soft);

  p {
    margin: 8px 0 0;
    color: var(--muted);
    line-height: 1.65;
  }
`

type EmptyStateProps = {
  text: string
}

export function EmptyState(props: EmptyStateProps) {
  return (
    <Root>
      <p>{props.text}</p>
    </Root>
  )
}
