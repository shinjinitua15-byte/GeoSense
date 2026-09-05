import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine():
    """Create and return a PostgreSQL/PostGIS database engine."""

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "geosense_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")

    connection_string = (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{name}"
    )

    return create_engine(connection_string)


def test_connection():
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT PostGIS_Version();")
        )

        print(
            "Connected! PostGIS version:",
            result.fetchone()[0]
        )


if __name__ == "__main__":
    test_connection()