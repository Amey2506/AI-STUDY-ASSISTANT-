from sqlalchemy import create_engine, inspect
import os

db_path = "study_assistant.db"
if os.path.exists(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():
        print(f"Table: {table_name}")
        for column in inspector.get_columns(table_name):
            print(f"  Column: {column['name']}")
else:
    print("Database file not found.")
