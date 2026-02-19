# Research: Infrastructure Overhaul

## Objectives
- Dockerize the environment.
- Remove `venv` dependency.
- Create a `Makefile` for ease of use.

## Findings
- Existing `docker-compose.yml` was messy and used inconsistent naming.
- `Dockerfile.api-dev` was using `requirements.txt` but could be optimized.
- `.dockerignore` was potentially leaking `node_modules` into the build context.
- No `Makefile` existed.

## Strategy
- Rewrite `docker-compose.yml` with explicit networks, volumes, and healthchecks.
- Create a `Makefile` to abstract `docker-compose` commands.
- Update `.dockerignore` to strictly exclude local artifacts.
