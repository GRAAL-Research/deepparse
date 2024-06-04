from pydantic import BaseModel


class Address(BaseModel):
    raw: str
