from contextlib import asynccontextmanager

from fastapi import FastAPI

from starlette.responses import StreamingResponse

from internal import interface
from internal.controller.http.handler.interview.model import *
from internal.controller.http.handler.vacancy.model import *


def NewHTTP(
        db: interface.IDB,
        vacancy_controller: interface.IVacancyController,
        interview_controller: interface.IInterviewController,
        telegram_controller: interface.ITelegramHTTPController,
        telegram_client: interface.ITelegramClient,
        http_middleware: interface.IHttpMiddleware,
        prefix: str
):
    app = FastAPI(
        openapi_url=prefix + "/openapi.json",
        docs_url=prefix + "/docs",
        redoc_url=prefix + "/redoc",
        lifespan=on_startup(telegram_client)
    )
    include_middleware(app, http_middleware)
    include_db_handler(app, db, prefix)

    include_vacancy_handlers(app, vacancy_controller, prefix)
    include_interview_handlers(app, interview_controller, prefix)
    include_telegram_handlers(app, telegram_controller, prefix)

    return app


def on_startup(telegram_client: interface.ITelegramClient):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await telegram_client.start()
        yield

    return lifespan


def include_middleware(
        app: FastAPI,
        http_middleware: interface.IHttpMiddleware
):
    http_middleware.authorization_middleware03(app)
    http_middleware.logger_middleware02(app)
    http_middleware.trace_middleware01(app)


def include_vacancy_handlers(
        app: FastAPI,
        vacancy_controller: interface.IVacancyController,
        prefix: str
):
    # Создание вакансии
    app.add_api_route(
        prefix + "/create",
        vacancy_controller.create_vacancy,
        tags=["Vacancy"],
        methods=["POST"],
    )

    # Удаление вакансии
    app.add_api_route(
        prefix + "/delete/{vacancy_id}",
        vacancy_controller.delete_vacancy,
        tags=["Vacancy"],
        methods=["DELETE"],
    )

    # Редактирование вакансии
    app.add_api_route(
        prefix + "/edit",
        vacancy_controller.edit_vacancy,
        tags=["Vacancy"],
        methods=["PUT"],
    )

    # Добавление вопроса к вакансии
    app.add_api_route(
        prefix + "/question/add",
        vacancy_controller.add_question,
        tags=["Question"],
        methods=["POST"],
    )

    # Редактирование вопроса
    app.add_api_route(
        prefix + "/question/edit",
        vacancy_controller.edit_question,
        tags=["Question"],
        methods=["PUT"],
    )

    # Удаление вопроса
    app.add_api_route(
        prefix + "/question/delete/{question_id}",
        vacancy_controller.delete_question,
        tags=["Question"],
        methods=["DELETE"],
    )

    # Создание весов критериев вакансии
    app.add_api_route(
        prefix + "/interview-weights/create",
        vacancy_controller.create_interview_weights,
        tags=["Vacancy"],
        methods=["POST"]
    )

    # Редактирование весов критериев вакансии
    app.add_api_route(
        prefix + "/interview-weights/edit",
        vacancy_controller.edit_interview_weights,
        tags=["Vacancy"],
        methods=["PUT"],
    )

    # Создание весов резюме
    app.add_api_route(
        prefix + "/resume-weights/create",
        vacancy_controller.create_resume_weights,
        tags=["Vacancy"],
        methods=["POST"],
    )

    # Редактирование весов резюме
    app.add_api_route(
        prefix + "/resume-weights/edit",
        vacancy_controller.edit_resume_weights,
        tags=["Vacancy"],
        methods=["PUT"],
    )

    # Генерация тегов
    app.add_api_route(
        prefix + "/generate-tags",
        vacancy_controller.generate_tags,
        methods=["POST"],
        tags=["Vacancy"],
        response_model=GenerateTagsResponse,
    )

    # Генерация вопросов
    app.add_api_route(
        prefix + "/question/generate",
        vacancy_controller.generate_question,
        methods=["POST"],
        tags=["Question"],
        response_model=GenerateQuestionResponse,
    )

    # Оценка резюме (множественная)
    app.add_api_route(
        prefix + "/evaluate-resumes",
        vacancy_controller.evaluate_resume,
        methods=["POST"],
        tags=["Resume"],
        response_model=EvaluateResumeResponse,
    )

    # Отклик на вакансию (одиночное резюме)
    app.add_api_route(
        prefix + "/respond",
        vacancy_controller.respond,
        methods=["POST"],
        tags=["Resume"],
        response_model=RespondResponse,
    )

    # Получение всех вакансий
    app.add_api_route(
        prefix + "/all",
        vacancy_controller.get_all_vacancy,
        methods=["GET"],
        tags=["Vacancy"],
        response_model=list[model.Vacancy],
    )

    # Получение всех вопросов вакансии
    app.add_api_route(
        prefix + "/question/all/{vacancy_id}",
        vacancy_controller.get_all_question,
        methods=["GET"],
        tags=["Question"],
        response_model=list[model.VacancyQuestion],
    )

    # Получение вопроса по ID
    app.add_api_route(
        prefix + "/question/{question_id}",
        vacancy_controller.get_question_by_id,
        methods=["GET"],
        tags=["Question"],
        response_model=model.VacancyQuestion,
    )

    # Получить веса оценки интервью
    app.add_api_route(
        prefix + "/interview-weights/{vacancy_id}",
        vacancy_controller.get_interview_weights,
        methods=["GET"],
        tags=["Vacancy"],
        response_model=list[model.InterviewWeights],
    )

    # Получение весов критериев резюме
    app.add_api_route(
        prefix + "/resume-weights/{vacancy_id}",
        vacancy_controller.get_resume_weights,
        methods=["GET"],
        tags=["Vacancy"],
        response_model=list[model.ResumeWeights],
    )


def include_interview_handlers(
        app: FastAPI,
        interview_controller: interface.IInterviewController,
        prefix: str
):
    # Начало интервью
    app.add_api_route(
        prefix + "/interview/start/{interview_id}",
        interview_controller.start_interview,
        methods=["POST"],
        tags=["Interview"],
        response_model=StartInterviewResponse,
    )

    # Отправка ответа на вопрос интервью
    app.add_api_route(
        prefix + "/interview/answer",
        interview_controller.send_answer,
        methods=["POST"],
        tags=["Interview"],
        response_model=SendAnswerResponse,
    )

    # Получение всех интервью для вакансии
    app.add_api_route(
        prefix + "/interview/vacancy/{vacancy_id}",
        interview_controller.get_all_interview,
        methods=["GET"],
        tags=["Interview"],
        response_model=list[model.Interview],
    )

    # Получить интервью по ID
    app.add_api_route(
        prefix + "/interview/{interview_id}",
        interview_controller.get_interview_by_id,
        methods=["GET"],
        tags=["Interview"],
        response_model=model.Interview,
    )

    # Получение деталей интервью
    app.add_api_route(
        prefix + "/interview/{interview_id}/details",
        interview_controller.get_interview_details,
        methods=["GET"],
        tags=["Interview"],
        response_model=GetCandidateAnswersResponse,
    )

    # Скачать аудио файл
    app.add_api_route(
        prefix + "/interview/audio/{audio_fid}/{audio_filename}",
        interview_controller.download_audio,
        methods=["GET"],
        tags=["Interview"],
        response_class=StreamingResponse,
    )

    # Скачать резюме
    app.add_api_route(
        prefix + "/interview/resume/{resume_fid}/{resume_filename}",
        interview_controller.download_resume,
        methods=["GET"],
        tags=["Interview"],
        response_class=StreamingResponse,
    )


def include_telegram_handlers(
        app: FastAPI,
        telegram_controller: interface.ITelegramHTTPController,
        prefix: str
):
    # Сгенерировать QR Code
    app.add_api_route(
        prefix + "/telegram/qr/generate",
        telegram_controller.generate_qr_code,
        methods=["GET"],
        tags=["Telegram"],
        response_class=StreamingResponse,
    )

    # Проверка статуса QR кода
    app.add_api_route(
        prefix + "/telegram/qr/status",
        telegram_controller.check_qr_status,
        methods=["GET"],
        tags=["Telegram"],
    )

    # Запуск Telegram клиента
    app.add_api_route(
        prefix + "/telegram/start",
        telegram_controller.start_telegram_client,
        methods=["POST"],
        tags=["Telegram"],
    )


def include_db_handler(app: FastAPI, db: interface.IDB, prefix: str):
    app.add_api_route(prefix + "/table/create", create_table_handler(db), methods=["GET"])
    app.add_api_route(prefix + "/table/drop", drop_table_handler(db), methods=["GET"])


def create_table_handler(db: interface.IDB):
    async def create_table():
        try:
            await db.multi_query(model.create_all_tables_queries)
        except Exception as err:
            raise err

    return create_table


def drop_table_handler(db: interface.IDB):
    async def drop_table():
        try:
            await db.multi_query(model.drop_all_tables_queries)
        except Exception as err:
            raise err

    return drop_table
