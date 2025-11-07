from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from internal import interface
from pkg.log_wrapper import auto_log
from pkg.trace_wrapper import traced_method


class TelegramHTTPController(interface.ITelegramHTTPController):
    def __init__(
            self,
            tel: interface.ITelemetry,
            telegram_client: interface.ITelegramClient,
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()
        self.telegram_client = telegram_client

    @auto_log()
    @traced_method()
    async def generate_qr_code(self) -> StreamingResponse:
        qr_image = await self.telegram_client.generate_qr_code()

        def iterfile():
            try:
                qr_image.seek(0)
                while True:
                    chunk = qr_image.read(8192)
                    if not chunk:
                        break
                    yield chunk
            finally:
                qr_image.close()

        return StreamingResponse(
            iterfile(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=telegram_qr.png",
                "Cache-Control": "no-cache"
            }
        )

    @auto_log()
    @traced_method()
    async def check_qr_status(self) -> JSONResponse:
        status, session_string = await self.telegram_client.qr_code_status()

        response_data = {
            "status": status,
            "session_string": session_string if status == "confirmed" else None
        }

        return JSONResponse(
            status_code=200,
            content=response_data
        )

    @auto_log()
    @traced_method()
    async def start_telegram_client(self) -> JSONResponse:
        await self.telegram_client.start()

        return JSONResponse(
            status_code=200,
            content={"message": "Telegram client started successfully"}
        )
