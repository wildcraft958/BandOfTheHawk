# Node 22 rather than 24 on purpose: npm 11 gates dependency install scripts
# behind an approval prompt, which silently skips esbuild's postinstall and
# breaks the Vite build inside a container. npm 10 does not.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s --retries=6 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
