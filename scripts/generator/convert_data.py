from sql.database import engine, SessionLocal
from sql.models import Base
from sql.schemas import ValDataRow

Base.metadata.create_all(bind=engine)

db = SessionLocal()
ros = db.query(ValDataRow).all()
