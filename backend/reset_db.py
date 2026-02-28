from database import Base, engine
from sqlalchemy import inspect

print("Dropping all tables...")
Base.metadata.drop_all(bind=engine)
print("Creating all tables...")
Base.metadata.create_all(bind=engine)

print("\nVerifying Schema:")
inspector = inspect(engine)
for table_name in inspector.get_table_names():
    print(f"Table: {table_name}")
    for column in inspector.get_columns(table_name):
        print(f"  Column: {column['name']}")
