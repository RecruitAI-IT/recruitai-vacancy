from abc import abstractmethod
from typing import Protocol

from fastapi import UploadFile, Form
from fastapi.responses import JSONResponse

from internal.controller.http.handler.vacancy.model import *


class IVacancyController(Protocol):
    @abstractmethod
    async def create_vacancy(self, body: CreateVacancyBody) -> JSONResponse: pass

    @abstractmethod
    async def delete_vacancy(self, vacancy_id: int) -> JSONResponse: pass

    @abstractmethod
    async def edit_vacancy(self, body: EditVacancyBody) -> JSONResponse: pass

    @abstractmethod
    async def add_question(self, body: AddQuestionBody) -> JSONResponse: pass

    @abstractmethod
    async def edit_question(self, body: EditQuestionBody) -> JSONResponse: pass

    @abstractmethod
    async def delete_question(self, question_id: int) -> JSONResponse: pass

    @abstractmethod
    async def create_interview_weights(self, body: CreateInterviewWeightsBody) -> JSONResponse: pass

    @abstractmethod
    async def edit_interview_weights(self, body: EditInterviewWeightsBody) -> JSONResponse: pass

    @abstractmethod
    async def create_resume_weights(self, body: CreateResumeWeightsBody) -> JSONResponse: pass

    @abstractmethod
    async def edit_resume_weights(self, body: EditResumeWeightsBody) -> JSONResponse: pass

    @abstractmethod
    async def generate_tags(self, body: GenerateTagsBody) -> JSONResponse: pass

    @abstractmethod
    async def generate_question(self, body: GenerateQuestionBody) -> JSONResponse: pass

    @abstractmethod
    async def evaluate_resume(
            self,
            vacancy_id: int = Form(...),
            candidate_resume_files: list[UploadFile] = Form(...)
    ) -> JSONResponse: pass

    @abstractmethod
    async def respond(
            self,
            vacancy_id: int = Form(...),
            candidate_email: str = Form(...),
            candidate_resume_file: UploadFile = Form(...)
    ) -> JSONResponse:
        pass

    @abstractmethod
    async def get_all_vacancy(self) -> JSONResponse: pass

    @abstractmethod
    async def get_all_question(self, vacancy_id: int) -> JSONResponse: pass

    @abstractmethod
    async def get_question_by_id(self, question_id: int) -> JSONResponse: pass

    @abstractmethod
    async def get_interview_weights(self, vacancy_id: int) -> JSONResponse:
        pass

    @abstractmethod
    async def get_resume_weights(self, vacancy_id: int) -> JSONResponse:
        pass


class IVacancyService(Protocol):
    @abstractmethod
    async def create_vacancy(
            self,
            name: str,
            tags: list[str],
            description: str,
            red_flags: str,
            skill_lvl: model.SkillLevel,
    ) -> int: pass

    @abstractmethod
    async def delete_vacancy(self, vacancy_id: int) -> None: pass

    @abstractmethod
    async def edit_vacancy(
            self,
            vacancy_id: int,
            name: str | None,
            tags: list[str] | None,
            description: str | None,
            red_flags: str | None,
            skill_lvl: model.SkillLevel | None,
    ) -> None: pass

    @abstractmethod
    async def add_question(
            self,
            vacancy_id: int,
            question: str,
            hint_for_evaluation: str,
            weight: int,
            question_type: model.QuestionsType,
            response_time: int,
    ) -> int: pass

    @abstractmethod
    async def edit_question(
            self,
            question_id: int,
            question: str | None,
            hint_for_evaluation: str | None,
            weight: int | None,
            question_type: model.QuestionsType | None,
            response_time: int | None,
    ) -> int: pass

    @abstractmethod
    async def delete_question(self, question_id: int) -> None: pass

    @abstractmethod
    async def create_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int,
            soft_skill_score_weight: int,
            hard_skill_score_weight: int,
            accordance_xp_resume_score_weight: int,
            accordance_skill_resume_score_weight: int,
            red_flag_score_weight: int,
    ) -> None: pass

    @abstractmethod
    async def edit_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int | None,
            soft_skill_score_weight: int | None,
            hard_skill_score_weight: int | None,
            accordance_xp_resume_score_weight: int | None,
            accordance_skill_resume_score_weight: int | None,
            red_flag_score_weight: int | None,
    ) -> None: pass

    @abstractmethod
    async def create_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int,
            accordance_skill_vacancy_score_threshold: int,
            recommendation_weight: int,
            portfolio_weight: int,
    ) -> None: pass

    @abstractmethod
    async def edit_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int | None,
            accordance_skill_vacancy_score_threshold: int | None,
            recommendation_weight: int | None,
            portfolio_weight: int | None,
    ) -> None: pass

    @abstractmethod
    async def generate_tags(self, vacancy_description: str) -> list[str]: pass

    @abstractmethod
    async def generate_question(
            self,
            vacancy_id: int,
            questions_type: model.QuestionsType,
            count_questions: int,
    ) -> list[model.VacancyQuestion]: pass

    @abstractmethod
    async def evaluate_resume(self, vacancy_id: int, candidate_resume_files: list[UploadFile]) -> list[
        model.Interview]: pass

    @abstractmethod
    async def respond(
            self,
            vacancy_id: int,
            candidate_email: str,
            candidate_resume_file: UploadFile
    ) -> tuple[str, int, int, str]:
        pass

    @abstractmethod
    async def get_all_vacancy(self) -> list[model.Vacancy]: pass

    @abstractmethod
    async def get_all_question(self, vacancy_id: int) -> list[model.VacancyQuestion]: pass

    @abstractmethod
    async def get_question_by_id(self, question_id: int) -> model.VacancyQuestion: pass

    @abstractmethod
    async def get_interview_weights(self, vacancy_id: int) -> list[model.InterviewWeights]:
        pass

    @abstractmethod
    async def get_resume_weights(self, vacancy_id: int) -> list[model.ResumeWeights]:
        pass


class IVacancyRepo(Protocol):
    @abstractmethod
    async def create_vacancy(
            self,
            name: str,
            tags: list[str],
            description: str,
            red_flags: str,
            skill_lvl: model.SkillLevel,
    ) -> int: pass

    @abstractmethod
    async def delete_vacancy(self, vacancy_id: int) -> None: pass

    @abstractmethod
    async def edit_vacancy(
            self,
            vacancy_id: int,
            name: str | None,
            tags: list[str] | None,
            description: str | None,
            red_flags: str | None,
            skill_lvl: model.SkillLevel | None,
    ) -> None: pass

    @abstractmethod
    async def add_question(
            self,
            vacancy_id: int,
            question: str,
            hint_for_evaluation: str,
            weight: int,
            question_type: model.QuestionsType,
            response_time: int,
    ) -> int: pass

    @abstractmethod
    async def edit_question(
            self,
            question_id: int,
            question: str | None,
            hint_for_evaluation: str | None,
            weight: int | None,
            question_type: model.QuestionsType | None,
            response_time: int | None,
    ) -> None: pass

    @abstractmethod
    async def delete_question(self, question_id: int) -> None: pass

    @abstractmethod
    async def create_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int,
            soft_skill_score_weight: int,
            hard_skill_score_weight: int,
            accordance_xp_resume_score_weight: int,
            accordance_skill_resume_score_weight: int,
            red_flag_score_weight: int,
    ) -> None: pass

    @abstractmethod
    async def edit_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int | None,
            soft_skill_score_weight: int | None,
            hard_skill_score_weight: int | None,
            accordance_xp_resume_score_weight: int | None,
            accordance_skill_resume_score_weight: int | None,
            red_flag_score_weight: int | None,
    ) -> None: pass

    @abstractmethod
    async def create_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int,
            accordance_skill_vacancy_score_threshold: int,
            recommendation_weight: int,
            portfolio_weight: int,
    ) -> None: pass

    @abstractmethod
    async def edit_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int | None,
            accordance_skill_vacancy_score_threshold: int | None,
            recommendation_weight: int | None,
            portfolio_weight: int | None,
    ) -> None: pass

    @abstractmethod
    async def get_vacancy_by_id(self, vacancy_id: int) -> list[model.Vacancy]: pass

    @abstractmethod
    async def get_all_vacancy(self) -> list[model.Vacancy]: pass

    @abstractmethod
    async def get_all_question(self, vacancy_id: int) -> list[model.VacancyQuestion]: pass

    @abstractmethod
    async def get_question_by_id(self, question_id: int) -> list[model.VacancyQuestion]: pass

    @abstractmethod
    async def get_interview_weights(self, vacancy_id: int) -> list[model.InterviewWeights]: pass

    @abstractmethod
    async def get_resume_weights(self, vacancy_id: int) -> list[model.ResumeWeights]: pass


class IVacancyPromptGenerator(Protocol):
    @abstractmethod
    def get_question_generation_prompt(
            self,
            vacancy: model.Vacancy,
            count_questions: int,
            questions_type: model.QuestionsType,
    ) -> str: pass

    @abstractmethod
    def get_resume_evaluation_system_prompt(
            self,
            vacancy_description: str,
            vacancy_red_flags: str,
            vacancy_name: str,
            vacancy_tags: list[str]
    ) -> str: pass

    @abstractmethod
    def get_generate_tags_system_prompt(self) -> str: pass
