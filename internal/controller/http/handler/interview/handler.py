from fastapi import UploadFile, Form, Path, File
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from internal import interface
from pkg.log_wrapper import auto_log
from pkg.trace_wrapper import traced_method


class InterviewController(interface.IInterviewController):
    def __init__(
            self,
            tel: interface.ITelemetry,
            interview_service: interface.IInterviewService,
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()
        self.interview_service = interview_service

    @auto_log()
    @traced_method()
    async def start_interview(self, interview_id: int) -> JSONResponse:

        message_to_candidate, total_questions, question_id, llm_audio_filename, llm_audio_fid = await self.interview_service.start_interview(
            interview_id
        )

        return JSONResponse(
            status_code=200,
            content={
                "message_to_candidate": message_to_candidate,
                "total_question": total_questions,
                "question_id": question_id,
                "llm_audio_filename": llm_audio_filename,
                "llm_audio_fid": llm_audio_fid,
            }
        )

    @auto_log()
    @traced_method()
    async def send_answer(
            self,
            interview_id: int = Form(...),
            question_id: int = Form(...),
            audio_file: UploadFile = File(...)
    ) -> JSONResponse:
        next_question_id, message_to_candidate, interview_result, llm_audio_filename, llm_audio_fid = await self.interview_service.send_answer(
            interview_id=interview_id,
            question_id=question_id,
            audio_file=audio_file
        )
        return JSONResponse(
            status_code=200,
            content={
                "question_id": next_question_id,
                "message_to_candidate": message_to_candidate,
                "interview_result": interview_result,
                "llm_audio_filename": llm_audio_filename,
                "llm_audio_fid": llm_audio_fid,
            }
        )

    @auto_log()
    @traced_method()
    async def get_all_interview(self, vacancy_id: int = Path(...)) -> JSONResponse:

        interviews = await self.interview_service.get_all_interview(vacancy_id)
        interviews_dict = [interview.to_dict() for interview in interviews]

        return JSONResponse(
            status_code=200,
            content=interviews_dict
        )

    @auto_log()
    @traced_method()
    async def get_interview_by_id(self, interview_id: int = Path(...)) -> JSONResponse:
        interview = await self.interview_service.get_interview_by_id(interview_id)
        interview_dict = interview.to_dict()

        return JSONResponse(
            status_code=200,
            content=interview_dict
        )

    @auto_log()
    @traced_method()
    async def get_interview_details(self, interview_id: int = Path(...)) -> JSONResponse:
        candidate_answers, interview_messages = await self.interview_service.get_interview_details(
            interview_id
        )
        candidate_answers_dict = [answer.to_dict() for answer in candidate_answers]
        interview_messages_dict = [message.to_dict() for message in interview_messages]

        return JSONResponse(
            status_code=200,
            content={
                "candidate_answers": candidate_answers_dict,
                "interview_messages": interview_messages_dict
            }
        )

    @auto_log()
    @traced_method()
    async def download_audio(
            self,
            audio_fid: str = Path(...),
            audio_filename: str = Path(...)
    ) -> StreamingResponse:
        audio_stream, content_type = await self.interview_service.download_audio(audio_fid, audio_filename)

        # Определяем MIME тип для аудио файлов
        if not content_type or content_type == "application/octet-stream":
            if audio_filename.lower().endswith(('.mp3', '.mpeg')):
                content_type = "audio/mpeg"
            elif audio_filename.lower().endswith('.wav'):
                content_type = "audio/wav"
            elif audio_filename.lower().endswith('.ogg'):
                content_type = "audio/ogg"
            elif audio_filename.lower().endswith('.m4a'):
                content_type = "audio/mp4"
            else:
                content_type = "audio/mpeg"  # default

        def iterfile():
            try:
                while True:
                    chunk = audio_stream.read(8192)  # Читаем по 8KB
                    if not chunk:
                        break
                    yield chunk
            finally:
                audio_stream.close()

        return StreamingResponse(
            iterfile(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={audio_filename}",
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes"
            }
        )

    @auto_log()
    @traced_method()
    async def download_resume(
            self,
            resume_fid: str = Path(...),
            resume_filename: str = Path(...)
    ) -> StreamingResponse:
        # Получаем поток файла резюме из сервиса
        resume_stream, content_type = await self.interview_service.download_resume(
            resume_fid,
            resume_filename
        )

        # Определяем MIME тип для резюме
        if not content_type or content_type == "application/octet-stream":
            if resume_filename.lower().endswith('.pdf'):
                content_type = "application/pdf"
            elif resume_filename.lower().endswith(('.doc', '.docx')):
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif resume_filename.lower().endswith('.txt'):
                content_type = "text/plain"
            else:
                content_type = "application/octet-stream"  # default

        def iterfile():
            try:
                while True:
                    chunk = resume_stream.read(8192)  # Читаем по 8KB
                    if not chunk:
                        break
                    yield chunk
            finally:
                resume_stream.close()

        return StreamingResponse(
            iterfile(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={resume_filename}",
                "Cache-Control": "no-cache",
                "Accept-Ranges": "bytes"
            }
        )
