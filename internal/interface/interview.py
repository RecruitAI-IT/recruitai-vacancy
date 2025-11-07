import io
from abc import abstractmethod
from typing import Protocol

from fastapi import UploadFile, Form, File, Path
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from internal import model


class IInterviewController(Protocol):
    @abstractmethod
    async def start_interview(self, interview_id: int) -> JSONResponse:
        pass

    @abstractmethod
    async def send_answer(
            self,
            interview_id: int = Form(...),
            question_id: int = Form(...),
            audio_file: UploadFile = File(...)
    ) -> JSONResponse:
        pass

    @abstractmethod
    async def get_all_interview(self, vacancy_id: int) -> JSONResponse: pass

    @abstractmethod
    async def get_interview_by_id(self, interview_id: int) -> JSONResponse:
        pass

    @abstractmethod
    async def get_interview_details(self, interview_id: int) -> JSONResponse: pass

    @abstractmethod
    async def download_audio(
            self,
            audio_fid: str = Path(...),
            audio_filename: str = Path(...)
    ) -> StreamingResponse: pass

    async def download_resume(
            self,
            resume_fid: str = Path(...),
            resume_filename: str = Path(...)
    ) -> StreamingResponse: pass


class IInterviewService(Protocol):
    @abstractmethod
    async def start_interview(self, interview_id: int) -> tuple[str, int, int, str, str]:
        pass

    @abstractmethod
    async def send_answer(
            self,
            interview_id: int,
            question_id: int,
            audio_file: UploadFile
    ) -> tuple[int, str, dict, str, str]:
        pass

    async def get_all_interview(self, vacancy_id: int) -> list[model.Interview]: pass

    @abstractmethod
    async def get_interview_by_id(self, interview_id: int) -> model.Interview:
        pass

    @abstractmethod
    async def get_interview_details(
            self,
            interview_id: int
    ) -> tuple[list[model.CandidateAnswer], list[model.InterviewMessage]]: pass

    @abstractmethod
    async def download_audio(self, audio_fid: str, audio_filename: str) -> tuple[io.BytesIO, str]: pass

    @abstractmethod
    async def download_resume(self, resume_fid: str, resume_filename: str) -> tuple[io.BytesIO, str]: pass


class IInterviewRepo(Protocol):
    @abstractmethod
    async def create_interview(
            self,
            vacancy_id: int,
            candidate_name: str,
            candidate_email: str,
            candidate_phone: str,
            candidate_telegram_login: str,
            candidate_resume_fid: str,
            candidate_resume_filename: str,
            accordance_xp_vacancy_score: int,
            accordance_skill_vacancy_score: int,
    ) -> int:
        pass

    @abstractmethod
    async def create_candidate_answer(
            self,
            question_id: int,
            interview_id: int,
    ) -> int: pass

    @abstractmethod
    async def create_interview_message(
            self,
            interview_id: int,
            question_id: int,
            audio_name: str,
            audio_fid: str,
            role: str,
            text: str,
    ) -> int:
        pass

    @abstractmethod
    async def add_message_to_candidate_answer(
            self,
            message_id: int,
            candidate_answer_id: int
    ) -> None: pass

    @abstractmethod
    async def evaluation_candidate_answer(
            self,
            candidate_answer_id: int,
            score: int,
            message_to_candidate: str,
            message_to_hr: str,
            response_time: int
    ) -> None: pass

    @abstractmethod
    async def fill_interview_criterion(
            self,
            interview_id: int,
            red_flag_score: int,
            hard_skill_score: int,
            soft_skill_score: int,
            logic_structure_score: int,
            accordance_xp_resume_score: int,
            accordance_skill_resume_score: int,
            strong_areas: str,
            weak_areas: str,
            approved_skills: list[str],
            general_score: float,
            general_result: model.GeneralResult,
            message_to_candidate: str,
            message_to_hr: str,
    ) -> None:
        pass

    @abstractmethod
    async def get_candidate_answer(
            self,
            question_id: int,
            interview_id: int,
    ) -> list[model.CandidateAnswer]: pass

    @abstractmethod
    async def get_interview_by_id(self, interview_id: int) -> list[model.Interview]: pass

    @abstractmethod
    async def get_all_interview(self, vacancy_id: int) -> list[model.Interview]:
        pass

    @abstractmethod
    async def get_all_candidate_answer(self, interview_id: int) -> list[model.CandidateAnswer]:
        pass

    @abstractmethod
    async def get_interview_messages(self, interview_id: int) -> list[model.InterviewMessage]:
        pass


class IInterviewPromptGenerator(Protocol):
    @abstractmethod
    def get_hello_interview_system_prompt(
            self,
            vacancy: model.Vacancy,
            questions: list[model.VacancyQuestion],
            candidate_name: str
    ) -> str: pass

    @abstractmethod
    def get_interview_management_system_prompt(
            self,
            vacancy: model.Vacancy,
            questions: list[model.VacancyQuestion],
            current_question_order_number: int
    ) -> str: pass

    @abstractmethod
    def get_answer_evaluation_system_prompt(
            self,
            question: model.VacancyQuestion,
            vacancy: model.Vacancy
    ) -> str: pass

    @abstractmethod
    def get_interview_summary_system_prompt(
            self,
            vacancy: model.Vacancy,
            questions: list[model.VacancyQuestion],
    ) -> str: pass
