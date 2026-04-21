import { createBuilder } from './.modules/aspire.js'

const builder = await createBuilder()

await builder.addDockerComposeEnvironment('env')

const frontend = await builder
  .addViteApp('frontend', './web-app/frontend')
  .withNpm()
  .withExternalHttpEndpoints()

await builder.build().run()
