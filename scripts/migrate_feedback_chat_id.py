"""Add the feedback-to-chat link without modifying existing feedback rows."""

from __future__ import annotations

import argparse

from sqlalchemy import inspect, text

from app.db.database import engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the schema change")
    args = parser.parse_args()

    if engine.dialect.name not in {"postgresql", "sqlite"}:
        raise RuntimeError("This migration supports PostgreSQL and SQLite only.")

    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        if not {"feedback", "chat_history"} <= tables:
            raise RuntimeError("Required tables are missing. Initialise the database first.")

        columns = {column["name"] for column in inspector.get_columns("feedback")}
        has_column = "chat_id" in columns
        if has_column:
            foreign_keys = inspector.get_foreign_keys("feedback")
            linked = any(
                fk["constrained_columns"] == ["chat_id"]
                and fk["referred_table"] == "chat_history"
                and fk["referred_columns"] == ["id"]
                for fk in foreign_keys
            )
            if not linked:
                raise RuntimeError("feedback.chat_id exists without the expected foreign key.")

        has_index = any(
            index["column_names"] == ["chat_id"]
            for index in inspector.get_indexes("feedback")
        )
        if has_column and has_index:
            print("Feedback chat link is already migrated.")
            return

        if not args.apply:
            print("Pending: add nullable feedback.chat_id with a foreign key and index.")
            print("Existing feedback stays unchanged with chat_id = NULL.")
            print("Run again with --apply to migrate.")
            return

        if not has_column:
            connection.execute(
                text(
                    "ALTER TABLE feedback ADD COLUMN chat_id INTEGER "
                    "REFERENCES chat_history(id)"
                )
            )
        if not has_index:
            connection.execute(text("CREATE INDEX ix_feedback_chat_id ON feedback (chat_id)"))

    print("Migration complete: feedback.chat_id added with foreign key and index.")


if __name__ == "__main__":
    main()
