from dataclasses import dataclass

from telethon import TelegramClient
from telethon.sessions import StringSession


@dataclass
class QrSession:
    client: TelegramClient
    string_session: StringSession
    status: str

class QRCodeStatus:
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"
    ERROR = "error"