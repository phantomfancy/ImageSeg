FROM node:25-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
COPY web-app/contracts/package.json web-app/contracts/package.json
COPY web-app/frontend/package.json web-app/frontend/package.json

RUN npm ci
RUN npm install --no-save --workspace @4cimageseg/frontend @rolldown/binding-linux-x64-gnu@1.0.0-rc.15 lightningcss-linux-x64-gnu@1.32.0

COPY web-app/contracts web-app/contracts
COPY web-app/frontend web-app/frontend

RUN npm run build --workspace @4cimageseg/frontend

FROM nginx:1.29-alpine

COPY web-app/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web-app/frontend/dist /usr/share/nginx/html

EXPOSE 80
