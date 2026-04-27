FROM node:25-slim AS build
WORKDIR /app

COPY package.json package-lock.json ./
COPY imageseg-webapp/contracts/package.json imageseg-webapp/contracts/package.json
COPY imageseg-webapp/frontend/package.json imageseg-webapp/frontend/package.json

RUN npm ci
RUN npm install --no-save --workspace @4cimageseg/frontend @rolldown/binding-linux-x64-gnu@1.0.0-rc.15 lightningcss-linux-x64-gnu@1.32.0

COPY imageseg-webapp/contracts imageseg-webapp/contracts
COPY imageseg-webapp/frontend imageseg-webapp/frontend

RUN npm run build --workspace @4cimageseg/frontend

FROM nginx:1.29-alpine

COPY imageseg-webapp/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/imageseg-webapp/frontend/dist /usr/share/nginx/html

EXPOSE 80
