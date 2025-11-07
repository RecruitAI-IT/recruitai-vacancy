import io
import uuid

from fastapi import UploadFile

from internal import model, interface
from pkg.trace_wrapper import traced_method


class InterviewService(interface.IInterviewService):
    def __init__(
            self,
            tel: interface.ITelemetry,
            vacancy_repo: interface.IVacancyRepo,
            interview_repo: interface.IInterviewRepo,
            interview_prompt_generator: interface.IInterviewPromptGenerator,
            openai_client: interface.IOpenAIClient,
            anthropic_client: interface.IAnthropicClient,
            storage: interface.IStorage
    ):
        self.logger = tel.logger()

        self.vacancy_repo = vacancy_repo
        self.interview_repo = interview_repo
        self.interview_prompt_generator = interview_prompt_generator
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.storage = storage

    @traced_method()
    async def start_interview(self, interview_id: int) -> tuple[str, int, int, str, str]:
        interview = (await self.interview_repo.get_interview_by_id(interview_id))[0]
        vacancy = (await self.vacancy_repo.get_vacancy_by_id(interview.vacancy_id))[0]
        questions = await self.vacancy_repo.get_all_question(vacancy.id)
        current_question = questions[0]

        hello_interview_system_prompt = self.interview_prompt_generator.get_hello_interview_system_prompt(
            vacancy,
            questions,
            interview.candidate_name,
        )

        history = [
            {"role": "user", "content": "Привет!"}
        ]


        hello_interview, _ = await self.anthropic_client.generate_json(
            history=history,
            system_prompt=hello_interview_system_prompt,
            llm_model="claude-haiku-4-5",
            temperature=1
        )

        message_to_candidate = hello_interview["message_to_candidate"]

        llm_audio = await self.openai_client.text_to_speech(message_to_candidate)
        llm_audio_filename = f"hello_interview_{interview_id}_{current_question.id}.mp3"
        llm_audio_file_io = io.BytesIO(llm_audio)
        upload_response = await self.storage.upload(llm_audio_file_io, llm_audio_filename)
        llm_audio_fid = upload_response.fid

        llm_message_id = await self.interview_repo.create_interview_message(
            interview_id=interview_id,
            question_id=current_question.id,
            audio_name=llm_audio_filename,
            audio_fid=llm_audio_fid,
            role="assistant",
            text=str(hello_interview),
        )

        candidate_answer_id = await self.interview_repo.create_candidate_answer(
            question_id=current_question.id,
            interview_id=interview_id,
        )

        await self.interview_repo.add_message_to_candidate_answer(
            message_id=llm_message_id,
            candidate_answer_id=candidate_answer_id
        )

        return message_to_candidate, len(questions), questions[0].id, llm_audio_filename, llm_audio_fid

    @traced_method()
    async def send_answer(
            self,
            interview_id: int,
            question_id: int,
            audio_file: UploadFile
    ) -> tuple[int, str, dict, str, str]:
        interview = (await self.interview_repo.get_interview_by_id(interview_id))[0]
        vacancy = (await self.vacancy_repo.get_vacancy_by_id(interview.vacancy_id))[0]
        questions = await self.vacancy_repo.get_all_question(vacancy.id)
        current_question_order_number = \
            [idx + 1 for idx, question in enumerate(questions) if question.id == question_id][0]
        current_question = questions[current_question_order_number - 1]
        candidate_answer = (await self.interview_repo.get_candidate_answer(question_id, interview_id))[0]

        # 2. Транскрибируем аудио
        audio_content = await audio_file.read()
        transcribed_text, _ = await self.openai_client.transcribe_audio(
            audio_content,
            audio_file.filename,
            "whisper-1",
            "ru"
        )

        # 3. Сохраняем аудио в storage
        audio_file_io = io.BytesIO(audio_content)
        upload_response = await self.storage.upload(audio_file_io, audio_file.filename)
        audio_fid = upload_response.fid

        # 4. Создаем сообщение от кандидата
        candidate_message_id = await self.interview_repo.create_interview_message(
            interview_id=interview_id,
            question_id=question_id,
            audio_name=audio_file.filename,
            audio_fid=audio_fid,
            role="user",
            text=transcribed_text
        )
        await self.interview_repo.add_message_to_candidate_answer(
            message_id=candidate_message_id,
            candidate_answer_id=candidate_answer.id
        )

        # 5. Определяем действие через LLM (delve_into_question, next_question, finish_interview)
        interview_management_system_prompt = self.interview_prompt_generator.get_interview_management_system_prompt(
            vacancy=vacancy,
            questions=questions,
            current_question_order_number=current_question_order_number
        )
        interview_messages = await self.interview_repo.get_interview_messages(interview_id)

        interview_history = []
        for msg in interview_messages:
            interview_history.append({
                "role": msg.role,
                "content": msg.text
            })

        llm_response_json, _ = await self.anthropic_client.generate_json(
            history=interview_history,
            system_prompt=interview_management_system_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-haiku-4-5",
            temperature=1
        )

        action = llm_response_json["action"]
        message_to_candidate: str = llm_response_json["message_to_candidate"]

        # 6. Создаем голосовое для кандидата
        llm_audio = await self.openai_client.text_to_speech(message_to_candidate)
        llm_audio_filename = f"message_to_candidate_{interview_id}_{current_question.id}_{uuid.uuid4()}.mp3"
        llm_audio_file_io = io.BytesIO(llm_audio)
        upload_response = await self.storage.upload(llm_audio_file_io, llm_audio_filename)
        llm_audio_fid = upload_response.fid

        # 7. Обрабатываем разные сценарии
        if action == "delve_into_question":
            self.logger.info("Углубление в вопрос")
            await self.__continue_question(
                candidate_answer_id=candidate_answer.id,
                interview_id=interview_id,
                question_id=question_id,
                llm_audio_filename=llm_audio_filename,
                llm_audio_fid=llm_audio_fid,
                llm_response_json=llm_response_json,
            )
            return (
                question_id,
                message_to_candidate,
                {},
                llm_audio_filename,
                llm_audio_fid
            )

        elif action == "next_question" and current_question_order_number < len(questions):
            self.logger.info("Переход к следующему вопросу")
            next_question = await self.__next_question(
                interview_id=interview_id,
                llm_audio_filename=llm_audio_filename,
                llm_audio_fid=llm_audio_fid,
                llm_response_json=llm_response_json,
                candidate_answer_id=candidate_answer.id,
                response_time=60,
                current_question=current_question,
                questions=questions,
                vacancy=vacancy,
                interview_messages=interview_messages,
            )
            return (
                next_question.id,
                message_to_candidate,
                {},
                llm_audio_filename,
                llm_audio_fid
            )

        elif action == "finish_interview" or action == "next_question" and current_question_order_number == len(
                questions):
            self.logger.info("Заканчиваем интервью")

            interview = await self.__finish_interview(
                interview_id=interview_id,
                candidate_answer_id=candidate_answer.id,
                response_time=60,
                llm_audio_filename=llm_audio_filename,
                llm_audio_fid=llm_audio_fid,
                llm_response_json=llm_response_json,
                questions=questions,
                interview_messages=interview_messages,
                vacancy=vacancy,
                current_question=current_question,
            )
            return (
                question_id,
                message_to_candidate,
                interview.to_dict(),
                llm_audio_filename,
                llm_audio_fid
            )

        return question_id, message_to_candidate, {}, llm_audio_filename, llm_audio_fid

    async def __continue_question(
            self,
            candidate_answer_id: int,
            interview_id: int,
            question_id: int,
            llm_audio_filename: str,
            llm_audio_fid: str,
            llm_response_json: dict,
    ):
        llm_message_id = await self.interview_repo.create_interview_message(
            interview_id=interview_id,
            question_id=question_id,
            audio_name=llm_audio_filename,
            audio_fid=llm_audio_fid,
            role="assistant",
            text=str(llm_response_json)
        )
        await self.interview_repo.add_message_to_candidate_answer(
            message_id=llm_message_id,
            candidate_answer_id=candidate_answer_id
        )

    async def __next_question(
            self,
            interview_id: int,
            llm_audio_filename: str,
            llm_audio_fid: str,
            llm_response_json: dict,
            candidate_answer_id: int,
            response_time: int,
            current_question: model.VacancyQuestion,
            questions: list[model.VacancyQuestion],
            vacancy: model.Vacancy,
            interview_messages: list[model.InterviewMessage],
    ) -> model.VacancyQuestion | None:
        await self.__evaluate_answer(
            candidate_answer_id=candidate_answer_id,
            response_time=response_time,
            current_question=current_question,
            vacancy=vacancy,
            interview_messages=interview_messages,
        )

        for i, question in enumerate(questions):
            if question.id == current_question.id and i + 1 < len(questions):
                next_question = questions[i + 1]

                next_candidate_message_id = await self.interview_repo.create_candidate_answer(
                    next_question.id,
                    interview_id
                )
                llm_message_id = await self.interview_repo.create_interview_message(
                    interview_id=interview_id,
                    question_id=next_question.id,
                    audio_name=llm_audio_filename,
                    audio_fid=llm_audio_fid,
                    role="assistant",
                    text=str(llm_response_json)
                )

                await self.interview_repo.add_message_to_candidate_answer(
                    message_id=llm_message_id,
                    candidate_answer_id=next_candidate_message_id
                )

                return next_question
        return None

    async def __finish_interview(
            self,
            interview_id: int,
            candidate_answer_id: int,
            response_time: int,
            llm_audio_filename: str,
            llm_audio_fid: str,
            llm_response_json: dict,
            questions: list[model.VacancyQuestion],
            interview_messages: list[model.InterviewMessage],
            vacancy: model.Vacancy,
            current_question: model.VacancyQuestion
    ) -> model.Interview:
        # Оцениваем ответ на последний вопрос
        llm_message_id = await self.interview_repo.create_interview_message(
            interview_id=interview_id,
            question_id=current_question.id,
            audio_name=llm_audio_filename,
            audio_fid=llm_audio_fid,
            role="assistant",
            text=str(llm_response_json)
        )

        await self.interview_repo.add_message_to_candidate_answer(
            message_id=llm_message_id,
            candidate_answer_id=candidate_answer_id
        )

        await self.__evaluate_answer(
            candidate_answer_id=candidate_answer_id,
            response_time=response_time,
            current_question=current_question,
            vacancy=vacancy,
            interview_messages=interview_messages,
        )

        # Подводим итоги интервью
        interview_summary_system_prompt = self.interview_prompt_generator.get_interview_summary_system_prompt(
            vacancy=vacancy,
            questions=questions
        )

        interview_history = []
        for msg in interview_messages:
            interview_history.append({
                "role": msg.role,
                "content": msg.text
            })

        interview_history = interview_history + [
            {"role": "user", "content": "Подведи итоги интервью"}
        ]

        interview_evaluation, _ = await self.anthropic_client.generate_json(
            history=interview_history,
            system_prompt=interview_summary_system_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-sonnet-4-5",
            temperature=1
        )

        interview_weights = (await self.vacancy_repo.get_interview_weights(vacancy.id))[0]

        # Рассчитываем общий балл
        general_score = self.__calculate_general_score(
            red_flag_score=interview_evaluation["red_flag_score"],
            hard_skill_score=interview_evaluation["hard_skill_score"],
            soft_skill_score=interview_evaluation["soft_skill_score"],
            logic_structure_score=interview_evaluation["logic_structure_score"],
            accordance_xp_resume_score=interview_evaluation["accordance_xp_resume_score"],
            accordance_skill_resume_score=interview_evaluation["accordance_skill_resume_score"],
            interview_weights=interview_weights
        )

        # Определяем результат на основе порогового значения
        if general_score >= 7:
            general_result = model.GeneralResult.NEXT
        elif general_score >= 5:
            general_result = model.GeneralResult.DISPUTABLE
        else:
            general_result = model.GeneralResult.REJECTED

        await self.interview_repo.fill_interview_criterion(
            interview_id=interview_id,
            red_flag_score=interview_evaluation["red_flag_score"],
            hard_skill_score=interview_evaluation["hard_skill_score"],
            soft_skill_score=interview_evaluation["soft_skill_score"],
            logic_structure_score=interview_evaluation["logic_structure_score"],
            accordance_xp_resume_score=interview_evaluation["accordance_xp_resume_score"],
            accordance_skill_resume_score=interview_evaluation["accordance_skill_resume_score"],
            strong_areas=interview_evaluation["strong_areas"],
            weak_areas=interview_evaluation["weak_areas"],
            approved_skills=interview_evaluation["approved_skills"],
            general_score=general_score,
            general_result=general_result,
            message_to_candidate=interview_evaluation["message_to_candidate"],
            message_to_hr=interview_evaluation["message_to_hr"],
        )

        interview_data = await self.interview_repo.get_interview_by_id(interview_id)
        return interview_data[0]

    async def __evaluate_answer(
            self,
            candidate_answer_id: int,
            response_time: int,
            current_question: model.VacancyQuestion,
            vacancy: model.Vacancy,
            interview_messages: list[model.InterviewMessage]
    ) -> None:
        answer_evaluation_system_prompt = self.interview_prompt_generator.get_answer_evaluation_system_prompt(
            question=current_question,
            vacancy=vacancy
        )

        question_history = []
        for msg in interview_messages:
            if msg.question_id == current_question.id:
                question_history.append({
                    "role": msg.role,
                    "content": msg.text
                })

        question_history = question_history + [
            {"role": "user", "content": "Оцени мой ответ"}
        ]

        evaluation_data, _ = await self.anthropic_client.generate_json(
            history=question_history,
            system_prompt=answer_evaluation_system_prompt,
            max_tokens=20000,
            thinking_tokens=15000,
            llm_model="claude-sonnet-4-5",
            temperature=1
        )

        score = evaluation_data["score"]
        message_to_candidate = evaluation_data["message_to_candidate"]
        message_to_hr = evaluation_data["message_to_hr"]

        await self.interview_repo.evaluation_candidate_answer(
            candidate_answer_id=candidate_answer_id,
            score=score,
            message_to_hr=message_to_hr,
            message_to_candidate=message_to_candidate,
            response_time=response_time
        )

    def __calculate_general_score(
            self,
            red_flag_score: int,
            hard_skill_score: int,
            soft_skill_score: int,
            logic_structure_score: int,
            accordance_xp_resume_score: int,
            accordance_skill_resume_score: int,
            interview_weights: model.InterviewWeights
    ) -> int:
        # Получаем веса из объекта VacancyWeights
        w_red_flag = interview_weights.red_flag_score_weight
        w_hard_skill = interview_weights.hard_skill_score_weight
        w_soft_skill = interview_weights.soft_skill_score_weight
        w_logic_structure = interview_weights.logic_structure_score_weight
        w_accordance_xp = interview_weights.accordance_xp_resume_score_weight
        w_accordance_skill = interview_weights.accordance_skill_resume_score_weight

        # Рассчитываем общую сумму весов
        total_weight = (w_red_flag + w_hard_skill + w_soft_skill +
                        w_logic_structure + w_accordance_xp + w_accordance_skill)

        # Избегаем деления на ноль
        if total_weight == 0:
            return 0

        # Красные флаги работают в обратную сторону - чем больше красных флагов, тем хуже
        # Инвертируем red_flag_score (5 - red_flag_score)
        inverted_red_flag_score = 5 - red_flag_score

        # Рассчитываем взвешенную сумму
        weighted_sum = (
                inverted_red_flag_score * w_red_flag +
                hard_skill_score * w_hard_skill +
                soft_skill_score * w_soft_skill +
                logic_structure_score * w_logic_structure +
                accordance_xp_resume_score * w_accordance_xp +
                accordance_skill_resume_score * w_accordance_skill
        )

        # Максимально возможная оценка (все критерии по 5 баллов)
        max_possible_score = 5 * total_weight

        # Нормализуем к диапазону 0.0 - 1.0
        general_score = weighted_sum / max_possible_score

        # Ограничиваем значение диапазоном [0.0, 1.0]
        return int(max(0.0, min(1.0, general_score)) * 10)

    @traced_method()
    async def get_all_interview(self, vacancy_id: int) -> list[model.Interview]:
        return await self.interview_repo.get_all_interview(vacancy_id)

    @traced_method()
    async def get_interview_by_id(self, interview_id: int) -> model.Interview:
        try:
            interview = (await self.interview_repo.get_interview_by_id(interview_id))[0]

            return interview

        except Exception as err:
            raise err

    @traced_method()
    async def get_interview_details(
            self,
            interview_id: int
    ) -> tuple[list[model.CandidateAnswer], list[model.InterviewMessage]]:
        candidate_answers = await self.interview_repo.get_all_candidate_answer(interview_id)
        interview_messages = await self.interview_repo.get_interview_messages(interview_id)

        return candidate_answers, interview_messages

    @traced_method()
    async def download_audio(self, audio_fid: str, audio_filename: str) -> tuple[io.BytesIO, str]:
        try:
            audio_stream, content_type = await self.storage.download(audio_fid, audio_filename)

            return audio_stream, content_type

        except Exception as err:
            raise err

    @traced_method()
    async def download_resume(self, resume_fid: str, resume_filename: str) -> tuple[io.BytesIO, str]:
        try:
            audio_stream, content_type = await self.storage.download(resume_fid, resume_filename)
            return audio_stream, content_type

        except Exception as err:
            raise err
