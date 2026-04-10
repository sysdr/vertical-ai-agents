# Lesson 54 Cleanup + Run Guide

This directory contains setup and utility scripts for Lesson 54.

## Start the app

```bash
./start.sh
```

## Stop the app

```bash
./l54-agent-personalization/scripts/stop.sh
```

## Full cleanup

Stops local services, stops Docker compose, and prunes unused Docker resources:

```bash
./cleanup.sh
```

## Rebuild dependencies after cleanup

```bash
bash ./l54-agent-personalization/scripts/build.sh
```
