from sqlalchemy import Column, Integer, String, Boolean
from src.sql.database import Base


class TextModel(Base):
    __tablename__ = "val_data"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(String)
    label = Column(Boolean)
    prompt = Column(String, nullable=True)
    data_source = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    model_params = Column(String, nullable=True)

    persona_profile = Column(String, nullable=True)
    persona_mood = Column(String, nullable=True)
    person_ton = Column(String, nullable=True)

    augmentations = Column(String, nullable=True)
