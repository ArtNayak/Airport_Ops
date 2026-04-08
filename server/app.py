"""ASGI entry point used by OpenEnv multi-mode deployment validation."""

from app import app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
