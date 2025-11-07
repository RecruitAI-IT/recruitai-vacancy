from abc import abstractmethod
from typing import Protocol

from internal import model


class IRecruitAIAuthorizationClient(Protocol):
    @abstractmethod
    async def authorization(self, account_id: int) -> model.JWTTokens: pass

    @abstractmethod
    async def check_authorization(self, access_token: str) -> model.AuthorizationData: pass