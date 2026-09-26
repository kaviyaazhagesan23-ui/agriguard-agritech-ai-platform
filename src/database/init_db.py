from .db import database_url, init_database, make_engine


def main() -> None:
    engine = make_engine()
    init_database(engine)
    print(f"PaddyWise database tables are ready: {database_url()}")


if __name__ == "__main__":
    main()
