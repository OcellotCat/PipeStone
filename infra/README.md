# PipeStone monitoring

The stack is started from the repository root:

```bash
docker compose up --build -d
```

Endpoints:

- application: http://localhost:8000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3001
- raw application metrics: http://localhost:8000/metrics

Grafana uses `admin` / `admin` by default. Set `GRAFANA_ADMIN_USER` and
`GRAFANA_ADMIN_PASSWORD` before starting the stack in a non-local environment.
Host ports can be overridden with `APP_PORT`, `PROMETHEUS_PORT`, and
`GRAFANA_PORT`; the defaults are `8000`, `9091`, and `3001` respectively.

Copy `server_config.example.json` to `server_config.local.json` and fill in the
Telegram bot token and destination chat ID. The local configuration is mounted
as a Docker secret and is excluded from Git and the Docker build context.
