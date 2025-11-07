import asyncio
import io
from datetime import datetime
from fastapi import UploadFile

from internal import interface, model
from pkg.trace_wrapper import traced_method


class VacancyService(interface.IVacancyService):
    def __init__(
            self,
            tel: interface.ITelemetry,
            vacancy_repo: interface.IVacancyRepo,
            interview_repo: interface.IInterviewRepo,
            storage: interface.IStorage,
            vacancy_prompt_generator: interface.IVacancyPromptGenerator,
            anthropic_client: interface.IAnthropicClient,
            email_client: interface.IEmailClient,
            telegram_client: interface.ITelegramClient,
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()
        self.vacancy_repo = vacancy_repo
        self.interview_repo = interview_repo
        self.storage = storage
        self.vacancy_prompt_generator = vacancy_prompt_generator
        self.anthropic_client = anthropic_client
        self.email_client = email_client
        self.telegram_client = telegram_client

    @traced_method()
    async def create_vacancy(
            self,
            name: str,
            tags: list[str],
            description: str,
            red_flags: str,
            skill_lvl: model.SkillLevel,
    ) -> int:

        vacancy_id = await self.vacancy_repo.create_vacancy(
            name=name,
            tags=tags,
            description=description,
            red_flags=red_flags,
            skill_lvl=skill_lvl
        )
        return vacancy_id

    @traced_method()
    async def delete_vacancy(self, vacancy_id: int) -> None:
        await self.vacancy_repo.delete_vacancy(vacancy_id)

    @traced_method()
    async def edit_vacancy(
            self,
            vacancy_id: int,
            name: str | None,
            tags: list[str] | None,
            description: str | None,
            red_flags: str | None,
            skill_lvl: model.SkillLevel | None,
    ) -> None:
        await self.vacancy_repo.edit_vacancy(
            vacancy_id=vacancy_id,
            name=name,
            tags=tags,
            description=description,
            red_flags=red_flags,
            skill_lvl=skill_lvl
        )

    @traced_method()
    async def add_question(
            self,
            vacancy_id: int,
            question: str,
            hint_for_evaluation: str,
            weight: int,
            question_type: model.QuestionsType,
            response_time: int,
    ) -> int:
        question_id = await self.vacancy_repo.add_question(
            vacancy_id=vacancy_id,
            question=question,
            hint_for_evaluation=hint_for_evaluation,
            weight=weight,
            question_type=question_type,
            response_time=response_time
        )

        return question_id

    @traced_method()
    async def edit_question(
            self,
            question_id: int,
            question: str | None,
            hint_for_evaluation: str | None,
            weight: int | None,
            question_type: model.QuestionsType | None,
            response_time: int | None,
    ) -> int:
        await self.vacancy_repo.edit_question(
            question_id=question_id,
            question=question,
            hint_for_evaluation=hint_for_evaluation,
            weight=weight,
            question_type=question_type,
            response_time=response_time
        )

        return question_id

    @traced_method()
    async def delete_question(self, question_id: int) -> None:

        await self.vacancy_repo.delete_question(question_id)

    @traced_method()
    async def create_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int,
            soft_skill_score_weight: int,
            hard_skill_score_weight: int,
            accordance_xp_resume_score_weight: int,
            accordance_skill_resume_score_weight: int,
            red_flag_score_weight: int,
    ) -> None:

        await self.vacancy_repo.create_interview_weights(
            vacancy_id=vacancy_id,
            logic_structure_score_weight=logic_structure_score_weight,
            soft_skill_score_weight=soft_skill_score_weight,
            hard_skill_score_weight=hard_skill_score_weight,
            accordance_xp_resume_score_weight=accordance_xp_resume_score_weight,
            accordance_skill_resume_score_weight=accordance_skill_resume_score_weight,
            red_flag_score_weight=red_flag_score_weight
        )

    @traced_method()
    async def edit_interview_weights(
            self,
            vacancy_id: int,
            logic_structure_score_weight: int | None,
            soft_skill_score_weight: int | None,
            hard_skill_score_weight: int | None,
            accordance_xp_resume_score_weight: int | None,
            accordance_skill_resume_score_weight: int | None,
            red_flag_score_weight: int | None,
    ) -> None:
        await self.vacancy_repo.edit_interview_weights(
            vacancy_id=vacancy_id,
            logic_structure_score_weight=logic_structure_score_weight,
            soft_skill_score_weight=soft_skill_score_weight,
            hard_skill_score_weight=hard_skill_score_weight,
            accordance_xp_resume_score_weight=accordance_xp_resume_score_weight,
            accordance_skill_resume_score_weight=accordance_skill_resume_score_weight,
            red_flag_score_weight=red_flag_score_weight
        )

    @traced_method()
    async def create_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int,
            accordance_skill_vacancy_score_threshold: int,
            recommendation_weight: int,
            portfolio_weight: int,
    ) -> None:
        await self.vacancy_repo.create_resume_weights(
            vacancy_id=vacancy_id,
            accordance_xp_vacancy_score_threshold=accordance_xp_vacancy_score_threshold,
            accordance_skill_vacancy_score_threshold=accordance_skill_vacancy_score_threshold,
            recommendation_weight=recommendation_weight,
            portfolio_weight=portfolio_weight
        )

    @traced_method()
    async def edit_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int | None,
            accordance_skill_vacancy_score_threshold: int | None,
            recommendation_weight: int | None,
            portfolio_weight: int | None,
    ) -> None:
        await self.vacancy_repo.edit_resume_weights(
            vacancy_id=vacancy_id,
            accordance_xp_vacancy_score_threshold=accordance_xp_vacancy_score_threshold,
            accordance_skill_vacancy_score_threshold=accordance_skill_vacancy_score_threshold,
            recommendation_weight=recommendation_weight,
            portfolio_weight=portfolio_weight
        )

    @traced_method()
    async def generate_tags(self, vacancy_description: str) -> list[str]:
        generation_tag_system_prompt = self.vacancy_prompt_generator.get_generate_tags_system_prompt()

        history = [
            {"role": "user",
             "content": f"Проанализируй описание вакансии и извлеки ключевые теги:\n\n{vacancy_description}"}
        ]

        tags_json, _ = await self.anthropic_client.generate_json(
            history=history,
            system_prompt=generation_tag_system_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-sonnet-4-5",
            temperature=1
        )

        tags = tags_json.get("tags", [])

        return tags

    @traced_method()
    async def generate_question(
            self,
            vacancy_id: int,
            questions_type: model.QuestionsType,
            count_questions: int,
    ) -> list[model.VacancyQuestion]:
        vacancy = (await self.vacancy_repo.get_vacancy_by_id(vacancy_id))[0]

        question_generation_prompt = self.vacancy_prompt_generator.get_question_generation_prompt(
            vacancy=vacancy,
            count_questions=count_questions,
            questions_type=questions_type
        )

        history = [
            {"role": "user", "content": "Сгенерируй вопросы для интервью согласно требованиям."}
        ]

        questions_data, _ = await self.anthropic_client.generate_json(
            history=history,
            system_prompt=question_generation_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-sonnet-4-5",
            temperature=1
        )

        questions = []
        for q_data in questions_data["questions"]:
            question = model.VacancyQuestion(
                id=0,
                vacancy_id=vacancy_id,
                question=q_data.get("question", "Нет вопроса"),
                hint_for_evaluation=q_data.get("hint_for_evaluation", "Нет подсказки"),
                weight=q_data.get("weight", 1),
                question_type=model.QuestionsType(q_data.get("question_type", questions_type.value)),
                response_time=q_data.get("response_time", 5),
                created_at=datetime.now()
            )
            questions.append(question)

        return questions

    @traced_method()
    async def evaluate_resume(self, vacancy_id: int, candidate_resume_files: list[UploadFile]) -> list[model.Interview]:
        # Получаем данные вакансии один раз для всех резюме
        vacancy = (await self.vacancy_repo.get_vacancy_by_id(vacancy_id))[0]
        resume_weights = (await self.vacancy_repo.get_resume_weights(vacancy_id))[0]

        system_prompt = self.vacancy_prompt_generator.get_resume_evaluation_system_prompt(
            vacancy_description=vacancy.description,
            vacancy_red_flags=vacancy.red_flags,
            vacancy_name=vacancy.name,
            vacancy_tags=vacancy.tags
        )

        # Этап 1: Параллельно читаем все файлы резюме
        read_tasks = [
            self._read_resume_file(resume_file)
            for resume_file in candidate_resume_files
        ]

        # Получаем содержимое всех файлов параллельно
        file_contents = await asyncio.gather(*read_tasks, return_exceptions=True)

        # Подготавливаем данные для обработки (фильтруем ошибки чтения)
        resumes_to_process = []
        read_errors = []

        for i, content in enumerate(file_contents):
            if isinstance(content, Exception):
                self.logger.error(f"Ошибка при чтении файла {candidate_resume_files[i].filename}", {
                    "error": str(content),
                    "filename": candidate_resume_files[i].filename
                })
                read_errors.append({
                    "filename": candidate_resume_files[i].filename,
                    "error": str(content)
                })
            else:
                resumes_to_process.append({
                    "file": candidate_resume_files[i],
                    "content": content
                })

        # Этап 2: Параллельно обрабатываем все резюме через LLM
        process_tasks = [
            self._process_resume_with_llm(
                resume_data=resume_data,
                vacancy_id=vacancy_id,
                system_prompt=system_prompt,
                resume_weights=resume_weights
            )
            for resume_data in resumes_to_process
        ]

        # Выполняем все LLM запросы параллельно
        results = await asyncio.gather(*process_tasks, return_exceptions=True)

        # Фильтруем успешные результаты и ошибки
        created_interviews = []
        process_errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Ошибка при обработке резюме {resumes_to_process[i]['file'].filename}", {
                    "error": str(result),
                    "filename": resumes_to_process[i]['file'].filename
                })
                process_errors.append({
                    "filename": resumes_to_process[i]['file'].filename,
                    "error": str(result)
                })
            elif result is not None:
                created_interviews.append(result)

        return created_interviews

    async def _read_resume_file(self, resume_file: UploadFile) -> bytes:
        try:
            content = await resume_file.read()
            await resume_file.seek(0)
            return content
        except Exception as e:
            self.logger.error(f"Ошибка при чтении файла {resume_file.filename}", {"error": str(e)})
            raise e

    async def _process_resume_with_llm(
            self,
            resume_data: dict,
            vacancy_id: int,
            system_prompt: str,
            resume_weights: model.ResumeWeights
    ) -> model.Interview | None:
        """Обрабатывает резюме через LLM и создает интервью при необходимости"""
        try:
            resume_file = resume_data["file"]
            resume_content = resume_data["content"]

            history = [
                {"role": "user", "content": "Оцени это резюме"}
            ]

            # Генерируем ответ от LLM
            evaluation_data, _ = await self.anthropic_client.generate_json(
                history=history,
                system_prompt=system_prompt,
                max_tokens=20000,
                thinking_tokens=15000,
                llm_model="claude-sonnet-4-5",
                temperature=1,
                pdf_file=resume_content
            )

            accordance_xp_vacancy_score = evaluation_data.get("accordance_xp_vacancy_score", 0)
            accordance_skill_vacancy_score = evaluation_data.get("accordance_skill_vacancy_score", 0)
            candidate_email = evaluation_data.get("candidate_email", "unknown@example.com")
            candidate_name = evaluation_data.get("candidate_name", "Unknown")
            candidate_telegram_login = evaluation_data.get("candidate_telegram_login", "Unknown")
            candidate_phone = evaluation_data.get("candidate_phone", "Unknown")
            message_to_candidate = evaluation_data.get("message_to_candidate", "")
            message_to_hr = evaluation_data.get("message_to_hr", "")

            if (accordance_xp_vacancy_score >= resume_weights.accordance_xp_vacancy_score_threshold
                    and accordance_skill_vacancy_score >= resume_weights.accordance_skill_vacancy_score_threshold):

                self.logger.info("Кандидат прошел анализ резюме", {
                    "vacancy_id": vacancy_id,
                    "accordance_xp_vacancy_score": accordance_xp_vacancy_score,
                    "accordance_skill_vacancy_score": accordance_skill_vacancy_score,
                    "candidate_telegram_login": candidate_telegram_login,
                    "candidate_name": candidate_name,
                    "candidate_phone": candidate_phone,
                })

                resume_file_io = io.BytesIO(resume_content)
                upload_result = await self.storage.upload(resume_file_io, resume_file.filename)
                candidate_resume_fid = upload_result.fid

                interview_id = await self.interview_repo.create_interview(
                    vacancy_id=vacancy_id,
                    candidate_name=candidate_name,
                    candidate_email=candidate_email,
                    candidate_phone=candidate_phone,
                    candidate_telegram_login=candidate_telegram_login,
                    candidate_resume_fid=candidate_resume_fid,
                    candidate_resume_filename=resume_file.filename,
                    accordance_xp_vacancy_score=accordance_xp_vacancy_score,
                    accordance_skill_vacancy_score=accordance_skill_vacancy_score
                )

                # Создаем объект Interview для возврата
                interview = model.Interview(
                    id=interview_id,
                    vacancy_id=vacancy_id,
                    candidate_name=candidate_name,
                    candidate_email=candidate_email,
                    candidate_phone=candidate_phone,
                    candidate_telegram_login=candidate_telegram_login,
                    candidate_resume_fid=candidate_resume_fid,
                    candidate_resume_filename=resume_file.filename,
                    accordance_xp_vacancy_score=accordance_xp_vacancy_score,
                    accordance_skill_vacancy_score=accordance_skill_vacancy_score,
                    red_flag_score=0,
                    hard_skill_score=0,
                    soft_skill_score=0,
                    logic_structure_score=0,
                    accordance_xp_resume_score=0,
                    accordance_skill_resume_score=0,
                    strong_areas="",
                    weak_areas="",
                    approved_skills=[],
                    general_score=0.0,
                    general_result=model.GeneralResult.IN_PROCESS,
                    message_to_candidate=message_to_candidate,
                    message_to_hr=message_to_hr,
                    created_at=datetime.now()
                )

                return interview
            else:
                self.logger.info("Кандидат не прошел анализ резюме", {
                    "vacancy_id": vacancy_id,
                    "accordance_xp_vacancy_score": accordance_xp_vacancy_score,
                    "accordance_skill_vacancy_score": accordance_skill_vacancy_score,
                    "candidate_telegram_login": candidate_telegram_login,
                    "candidate_name": candidate_name,
                    "candidate_phone": candidate_phone,
                })
                return None

        except Exception as err:
            raise err

    @traced_method()
    async def respond(
            self,
            vacancy_id: int,
            candidate_email: str,
            candidate_resume_file: UploadFile
    ) -> tuple[str, int, int, str]:
        vacancy = (await self.vacancy_repo.get_vacancy_by_id(vacancy_id))[0]

        system_prompt = self.vacancy_prompt_generator.get_resume_evaluation_system_prompt(
            vacancy_description=vacancy.description,
            vacancy_red_flags=vacancy.red_flags,
            vacancy_name=vacancy.name,
            vacancy_tags=vacancy.tags
        )

        resume_content = await candidate_resume_file.read()

        history = [
            model.InterviewMessage(
                id=0,
                interview_id=0,
                question_id=0,
                audio_name="",
                audio_fid="",
                role="user",
                text="Оцени это резюме любой ценой",
                created_at=datetime.now()
            )
        ]
        history = [
            {"role": "user", "content": "Оцени это резюме"}
        ]

        evaluation_data, _ = await self.anthropic_client.generate_json(
            history=history,
            system_prompt=system_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-sonnet-4-5",
            temperature=1,
            pdf_file=resume_content
        )

        accordance_xp_vacancy_score = evaluation_data.get("accordance_xp_vacancy_score", 0)
        accordance_skill_vacancy_score = evaluation_data.get("accordance_skill_vacancy_score", 0)
        candidate_email = evaluation_data.get("candidate_email", "unknown@example.com")
        candidate_name = evaluation_data.get("candidate_name", "Unknown")
        candidate_telegram_login = evaluation_data.get("candidate_telegram_login", "Unknown")
        candidate_phone = evaluation_data.get("candidate_phone", "Unknown")
        message_to_candidate = evaluation_data.get("message_to_candidate", "")
        message_to_hr = evaluation_data.get("message_to_hr", "")

        resume_weights = (await self.vacancy_repo.get_resume_weights(vacancy_id))[0]

        if (accordance_xp_vacancy_score >= resume_weights.accordance_xp_vacancy_score_threshold
                and accordance_skill_vacancy_score >= resume_weights.accordance_skill_vacancy_score_threshold):

            self.logger.info("Кандидат прошел анализ резюме", {
                "vacancy_id": vacancy_id,
                "accordance_xp_vacancy_score": accordance_xp_vacancy_score,
                "accordance_skill_vacancy_score": accordance_skill_vacancy_score,
                "candidate_telegram_login": candidate_telegram_login,
                "candidate_name": candidate_name,
                "candidate_phone": candidate_phone,
            })

            resume_file_io = io.BytesIO(resume_content)
            upload_result = await self.storage.upload(resume_file_io, candidate_resume_file.filename)
            candidate_resume_fid = upload_result.fid

            interview_id = await self.interview_repo.create_interview(
                vacancy_id=vacancy_id,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                candidate_phone=candidate_phone,
                candidate_telegram_login=candidate_telegram_login,
                candidate_resume_fid=candidate_resume_fid,
                candidate_resume_filename=candidate_resume_file.filename,
                accordance_xp_vacancy_score=accordance_xp_vacancy_score,
                accordance_skill_vacancy_score=accordance_skill_vacancy_score
            )

            # Генерируем ссылку на интервью
            interview_link = f"/interview/start/{interview_id}"

            email_sent = await self.__send_interview_invitation_to_email(
                candidate_email=candidate_email,
                candidate_name=candidate_name,
                vacancy_name=vacancy.name,
                interview_id=interview_id
            )
            await self.__send_interview_invitation_to_telegram(
                candidate_telegram_login=candidate_telegram_login,
                candidate_phone=candidate_phone,
                vacancy_name=vacancy.name,
                interview_id=interview_id,
                vacancy_id=vacancy_id,
                candidate_name=candidate_name
            )

            return interview_link, accordance_xp_vacancy_score, accordance_skill_vacancy_score, message_to_candidate

        else:
            self.logger.info("Кандидат не прошел анализ резюме", {
                "vacancy_id": vacancy_id,
                "accordance_xp_vacancy_score": accordance_xp_vacancy_score,
                "accordance_skill_vacancy_score": accordance_skill_vacancy_score,
                "candidate_telegram_login": candidate_telegram_login,
                "candidate_name": candidate_name,
                "candidate_phone": candidate_phone,
            })

            return "", accordance_xp_vacancy_score, accordance_skill_vacancy_score, message_to_candidate

    @traced_method()
    async def get_all_vacancy(self) -> list[model.Vacancy]:
        vacancies = await self.vacancy_repo.get_all_vacancy()
        return vacancies

    @traced_method()
    async def get_all_question(self, vacancy_id: int) -> list[model.VacancyQuestion]:
        questions = await self.vacancy_repo.get_all_question(vacancy_id)
        return questions

    @traced_method()
    async def get_question_by_id(self, question_id: int) -> model.VacancyQuestion:
        question = (await self.vacancy_repo.get_question_by_id(question_id))[0]
        return question

    @traced_method()
    async def get_interview_weights(self, vacancy_id: int) -> list[model.InterviewWeights]:
        weights = await self.vacancy_repo.get_interview_weights(vacancy_id)

        return weights

    @traced_method()
    async def get_resume_weights(self, vacancy_id: int) -> list[model.ResumeWeights]:
        weights = await self.vacancy_repo.get_resume_weights(vacancy_id)
        return weights

    async def __send_interview_invitation_to_email(
            self,
            candidate_email: str,
            candidate_name: str,
            vacancy_name: str,
            interview_id: int
    ) -> bool:
        try:
            subject = f"Приглашение на интервью - {vacancy_name}"

            body = f"""
<html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #2c3e50;">Приглашение на интервью</h2>

            <p>Здравствуйте, {candidate_name}!</p>

            <p>Поздравляем! Ваше резюме прошло предварительный отбор на позицию <strong>{vacancy_name}</strong>.</p>

            <p>Мы приглашаем Вас пройти следующий этап - интервью с нашей системой ИИ.</p>

            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                <h3 style="color: #495057; margin-top: 0;">Как пройти интервью:</h3>
                <ol>
                    <li>Перейдите по ссылке ниже</li>
                    <li>Следуйте инструкциям системы</li>
                    <li>Отвечайте на вопросы честно и подробно</li>
                    <li>Убедитесь, что у вас есть доступ к микрофону</li>
                </ol>
            </div>

            <div style="text-align: center; margin: 30px 0;">
                <a href="/interview/start/{interview_id}" 
                   style="background-color: #007bff; color: white; padding: 12px 30px; 
                          text-decoration: none; border-radius: 5px; display: inline-block;
                          font-weight: bold;">
                    Начать интервью
                </a>
            </div>

            <p style="color: #6c757d; font-size: 0.9em;">
                <strong>Важно:</strong> Интервью займет примерно 15-30 минут. 
                Рекомендуем проходить его в тихом месте с хорошим интернет-соединением.
            </p>

            <hr style="border: none; border-top: 1px solid #dee2e6; margin: 30px 0;">

            <p style="color: #6c757d; font-size: 0.8em;">
                Если у вас возникнут вопросы, пожалуйста, свяжитесь с нами.<br>
                Удачи на интервью!
            </p>
        </div>
    </body>
</html>
"""

            email_sent = await self.email_client.send_email(
                to_email=candidate_email,
                subject=subject,
                body=body,
                is_html=True
            )

            if email_sent:
                self.logger.info("Приглашение отправлено", {
                    "candidate_email": candidate_email,
                    "candidate_name": candidate_name,
                    "vacancy_name": vacancy_name,
                    "interview_id": interview_id
                })
            else:
                self.logger.error("Ошибка отправки приглашения", {
                    "candidate_email": candidate_email,
                    "candidate_name": candidate_name,
                    "vacancy_name": vacancy_name,
                    "interview_id": interview_id
                })

            return email_sent

        except Exception as err:
            return False

    async def __send_interview_invitation_to_telegram(
            self,
            candidate_telegram_login: str,
            candidate_phone: str,
            vacancy_name: str,
            vacancy_id: int,
            interview_id: int,
            candidate_name: str
    ) -> bool:
        interview_link = f"https://vtb-aihr.ru/vacancy/{vacancy_id}/interview/{interview_id}"

        message_text = f"""🎉 Поздравляем, {candidate_name}!

Ваше резюме прошло предварительный отбор на позицию:
📋 {vacancy_name}

Мы приглашаем вас пройти следующий этап - интервью с нашей AI-системой.

🔗 Ссылка для прохождения интервью:
{interview_link}

📝 Рекомендации:
- Убедитесь в наличии микрофона
- Выберите тихое место
- Интервью займет 15-30 минут
- Отвечайте честно и подробно

Удачи! 🍀"""

        try:
            if candidate_telegram_login == "Unknown":
                raise Exception("")

            await self.telegram_client.send_message_to_telegram(
                tg_user_data=candidate_telegram_login,
                text=message_text
            )
            telegram_sent = True

        except Exception as err:
            try:
                if candidate_phone == "Unknown":
                    raise Exception("")
                await self.telegram_client.send_message_to_telegram(
                    tg_user_data=candidate_phone,
                    text=message_text
                )
                telegram_sent = True
            except Exception as err:
                telegram_sent = False

        if telegram_sent:
            self.logger.info("Telegram уведомление отправлено успешно", {
                "candidate_telegram_login": candidate_telegram_login,
                "candidate_phone": candidate_phone
            })
            return True
        else:
            self.logger.warning("Telegram уведомление не отправлено", {
                "candidate_telegram_login": candidate_telegram_login,
                "candidate_phone": candidate_phone
            })
            return False
