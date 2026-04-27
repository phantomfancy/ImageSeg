import styled from 'styled-components'

const Root = styled.section`
  margin-top: 18px;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: var(--control-bg-soft);

  h3 {
    margin: 0 0 12px;
    font-size: 1rem;
  }
`

const List = styled.div`
  display: grid;
  gap: 10px;
`

const Row = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  border-radius: 14px;
  background: var(--soft-fill);

  code {
    font-family: var(--mono);
    font-size: 0.84rem;
  }

  @media (max-width: 560px) {
    align-items: flex-start;
    flex-direction: column;
  }
`

type SignatureTableProps = {
  title: string
  items: Array<{
    name: string
    dims: string
  }>
}

export function SignatureTable(props: SignatureTableProps) {
  return (
    <Root>
      <h3>{props.title}</h3>
      <List>
        {props.items.map((item) => (
          <Row key={`${props.title}-${item.name}`}>
            <span>{item.name}</span>
            <code>{item.dims}</code>
          </Row>
        ))}
      </List>
    </Root>
  )
}
