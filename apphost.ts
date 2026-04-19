import { createBuilder } from './.modules/aspire.js'

const builder = await createBuilder()

await builder
  .addViteApp('frontend', './web-app/frontend')
  .withNpm()

await builder.build().run()
