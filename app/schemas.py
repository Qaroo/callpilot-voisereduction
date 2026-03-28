from pydantic import BaseModel


class EnrollResponse(BaseModel):
    speaker_id: str
    message: str


class SpeakersResponse(BaseModel):
    speakers: list[str]
