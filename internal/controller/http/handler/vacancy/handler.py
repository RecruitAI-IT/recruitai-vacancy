from opentelemetry.trace import Status, StatusCode, SpanKind
from fastapi import UploadFile, Form
from fastapi.responses import JSONResponse

from pkg.log_wrapper import auto_log
from pkg.trace_wrapper import traced_method
from .model import *
from internal import interface


class VacancyController(interface.IVacancyController):
    def __init__(
            self,
            tel: interface.ITelemetry,
            vacancy_service: interface.IVacancyService,
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()
        self.vacancy_service = vacancy_service

    @auto_log()
    @traced_method()
    async def create_vacancy(self, body: CreateVacancyBody) -> JSONResponse:
        vacancy_id = await self.vacancy_service.create_vacancy(
            name=body.name,
            tags=body.tags,
            description=body.description,
            red_flags=body.red_flags,
            skill_lvl=body.skill_lvl
        )

        return JSONResponse(
            status_code=201,
            content={
                "message": "Vacancy created successfully",
                "vacancy_id": vacancy_id
            }
        )

    @auto_log()
    @traced_method()
    async def delete_vacancy(self, vacancy_id: int) -> JSONResponse:
        await self.vacancy_service.delete_vacancy(vacancy_id)

        return JSONResponse(
            status_code=200,
            content={"message": "Vacancy deleted successfully"}
        )

    @auto_log()
    @traced_method()
    async def edit_vacancy(self, body: EditVacancyBody) -> JSONResponse:
        await self.vacancy_service.edit_vacancy(
            vacancy_id=body.vacancy_id,
            name=body.name,
            tags=body.tags,
            description=body.description,
            red_flags=body.red_flags,
            skill_lvl=body.skill_lvl
        )

        return JSONResponse(
            status_code=200,
            content={"message": "Vacancy updated successfully"}
        )

    @auto_log()
    @traced_method()
    async def add_question(self, body: AddQuestionBody) -> JSONResponse:

        question_id = await self.vacancy_service.add_question(
            vacancy_id=body.vacancy_id,
            question=body.question,
            hint_for_evaluation=body.hint_for_evaluation,
            weight=body.weight,
            question_type=body.question_type,
            response_time=body.response_time
        )

        return JSONResponse(
            status_code=201,
            content={
                "message": "Question added successfully",
                "question_id": question_id
            }
        )

    @auto_log()
    @traced_method()
    async def edit_question(self, body: EditQuestionBody) -> JSONResponse:
        await self.vacancy_service.edit_question(
            question_id=body.question_id,
            question=body.question,
            hint_for_evaluation=body.hint_for_evaluation,
            weight=body.weight,
            question_type=body.question_type,
            response_time=body.response_time
        )

        return JSONResponse(
            status_code=200,
            content={"message": "Question updated successfully"}
        )

    @auto_log()
    @traced_method()
    async def delete_question(self, question_id: int) -> JSONResponse:
        await self.vacancy_service.delete_question(question_id)

        return JSONResponse(
            status_code=200,
            content={"message": "Question deleted successfully"}
        )

    async def create_interview_weights(self, body: CreateInterviewWeightsBody) -> JSONResponse:

        await self.vacancy_service.create_interview_weights(
            vacancy_id=body.vacancy_id,
            logic_structure_score_weight=body.logic_structure_score_weight,
            soft_skill_score_weight=body.soft_skill_score_weight,
            hard_skill_score_weight=body.hard_skill_score_weight,
            accordance_xp_resume_score_weight=body.accordance_xp_resume_score_weight,
            accordance_skill_resume_score_weight=body.accordance_skill_resume_score_weight,
            red_flag_score_weight=body.red_flag_score_weight
        )

        return JSONResponse(
            status_code=201,
            content={"message": "Vacancy criterion weights created successfully"}
        )

    @auto_log()
    @traced_method()
    async def edit_interview_weights(self, body: EditInterviewWeightsBody) -> JSONResponse:
        await self.vacancy_service.edit_interview_weights(
            vacancy_id=body.vacancy_id,
            logic_structure_score_weight=body.logic_structure_score_weight,
            soft_skill_score_weight=body.soft_skill_score_weight,
            hard_skill_score_weight=body.hard_skill_score_weight,
            accordance_xp_resume_score_weight=body.accordance_xp_resume_score_weight,
            accordance_skill_resume_score_weight=body.accordance_skill_resume_score_weight,
            red_flag_score_weight=body.red_flag_score_weight
        )

        return JSONResponse(
            status_code=200,
            content={"message": "Vacancy criterion weights updated successfully"}
        )

    @auto_log()
    @traced_method()
    async def create_resume_weights(self, body: CreateResumeWeightsBody) -> JSONResponse:
        await self.vacancy_service.create_resume_weights(
            vacancy_id=body.vacancy_id,
            accordance_xp_vacancy_score_threshold=body.accordance_xp_vacancy_score_threshold,
            accordance_skill_vacancy_score_threshold=body.accordance_skill_vacancy_score_threshold,
            recommendation_weight=body.recommendation_weight,
            portfolio_weight=body.portfolio_weight
        )

        return JSONResponse(
            status_code=201,
            content={"message": "Resume weights created successfully"}
        )

    @auto_log()
    @traced_method()
    async def edit_resume_weights(self, body: EditResumeWeightsBody) -> JSONResponse:
        await self.vacancy_service.edit_resume_weights(
            vacancy_id=body.vacancy_id,
            accordance_xp_vacancy_score_threshold=body.accordance_xp_vacancy_score_threshold,
            accordance_skill_vacancy_score_threshold=body.accordance_skill_vacancy_score_threshold,
            recommendation_weight=body.recommendation_weight,
            portfolio_weight=body.portfolio_weight
        )

        return JSONResponse(
            status_code=200,
            content={"message": "Resume weights updated successfully"}
        )

    @auto_log()
    @traced_method()
    async def generate_tags(self, body: GenerateTagsBody) -> JSONResponse:

        tags = await self.vacancy_service.generate_tags(body.vacancy_description)

        return JSONResponse(
            status_code=200,
            content={"tags": tags}
        )

    @auto_log()
    @traced_method()
    async def generate_question(self, body: GenerateQuestionBody) -> JSONResponse:
        if body.count_questions > 30:
            raise Exception("Too many questions")

        questions = await self.vacancy_service.generate_question(
            vacancy_id=body.vacancy_id,
            questions_type=body.questions_type,
            count_questions=body.count_questions
        )

        # Конвертируем в словари для JSON ответа
        questions_dict = [question.to_dict() for question in questions]

        return JSONResponse(
            status_code=200,
            content={"questions": questions_dict}
        )

    @auto_log()
    @traced_method()
    async def evaluate_resume(
            self,
            vacancy_id: int = Form(...),
            candidate_resume_files: list[UploadFile] = Form(...)
    ) -> JSONResponse:

        if len(candidate_resume_files) > 10:
            raise Exception("Too many files")

        created_interviews = await self.vacancy_service.evaluate_resume(
            vacancy_id=vacancy_id,
            candidate_resume_files=candidate_resume_files
        )

        # Формируем ответ согласно EvaluateResumeResponse
        evaluation_resumes = []
        for interview in created_interviews:
            evaluation_resumes.append({
                "candidate_email": interview.candidate_email,
                "candidate_name": interview.candidate_name,
                "candidate_phone": interview.candidate_phone,
                "accordance_xp_vacancy_score": interview.accordance_xp_vacancy_score,
                "accordance_skill_vacancy_score": interview.accordance_skill_vacancy_score
            })

        return JSONResponse(
            status_code=200,
            content={"evaluation_resumes": evaluation_resumes}
        )

    @auto_log()
    @traced_method()
    async def respond(
            self,
            vacancy_id: int = Form(...),
            candidate_email: str = Form(...),
            candidate_resume_file: UploadFile = Form(...)
    ) -> JSONResponse:
        interview_link, accordance_xp_score, accordance_skill_score, message_to_candidate = await self.vacancy_service.respond(
            vacancy_id=vacancy_id,
            candidate_email=candidate_email,
            candidate_resume_file=candidate_resume_file
        )

        response_data = {
            "interview_link": interview_link,
            "accordance_xp_vacancy_score": accordance_xp_score,
            "accordance_skill_vacancy_score": accordance_skill_score,
            "message_to_candidate": message_to_candidate
        }

        return JSONResponse(
            status_code=200,
            content=response_data
        )

    @auto_log()
    @traced_method()
    async def get_all_vacancy(self) -> JSONResponse:
        vacancies = await self.vacancy_service.get_all_vacancy()
        vacancies_dict = [vacancy.to_dict() for vacancy in vacancies]

        return JSONResponse(
            status_code=200,
            content=vacancies_dict
        )

    @auto_log()
    @traced_method()
    async def get_all_question(self, vacancy_id: int) -> JSONResponse:
        questions = await self.vacancy_service.get_all_question(vacancy_id)
        questions_dict = [question.to_dict() for question in questions]

        return JSONResponse(
            status_code=200,
            content=questions_dict
        )

    @auto_log()
    @traced_method()
    async def get_question_by_id(self, question_id: int) -> JSONResponse:
        question = await self.vacancy_service.get_question_by_id(question_id)
        question_dict = question.to_dict()

        return JSONResponse(
            status_code=200,
            content=question_dict
        )

    @auto_log()
    @traced_method()
    async def get_interview_weights(self, vacancy_id: int) -> JSONResponse:
        weights = await self.vacancy_service.get_interview_weights(vacancy_id)
        weights_dict = [weight.to_dict() for weight in weights]

        return JSONResponse(
            status_code=200,
            content=weights_dict
        )

    @auto_log()
    @traced_method()
    async def get_resume_weights(self, vacancy_id: int) -> JSONResponse:
        weights = await self.vacancy_service.get_resume_weights(vacancy_id)
        weights_dict = [weight.to_dict() for weight in weights]

        return JSONResponse(
            status_code=200,
            content=weights_dict
        )
