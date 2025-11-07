from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


@dataclass
class AsyncWeedOperationResponse:
    status_code: int
    content: bytes
    content_type: str
    headers: dict
    fid: Optional[str] = None
    url: Optional[str] = None
    size: Optional[int] = None

class AuthorizationData(BaseModel):
    account_id: int
    two_fa_status: bool
    role: str
    message: str
    status_code: int


class JWTTokens(BaseModel):
    access_token: str
    refresh_token: str
