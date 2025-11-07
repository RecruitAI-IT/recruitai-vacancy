from pkg.trace_wrapper import traced_method
from .sql_query import *
from internal import interface, model


class VacancyRepo(interface.IVacancyRepo):
    def __init__(self, tel: interface.ITelemetry, db: interface.IDB):
        self.db = db
        self.tracer = tel.tracer()

    @traced_method()
    async def create_vacancy(
            self,
            name: str,
            tags: list[str],
            description: str,
            red_flags: str,
            skill_lvl: model.SkillLevel,
    ) -> int:
        args = {
            'name': name,
            'tags': tags,
            'description': description,
            'red_flags': red_flags,
            'skill_lvl': skill_lvl.value,
        }
        vacancy_id = await self.db.insert(create_vacancy_query, args)

        return vacancy_id

    @traced_method()
    async def delete_vacancy(self, vacancy_id: int) -> None:

        args = {'vacancy_id': vacancy_id}
        await self.db.delete(delete_vacancy_query, args)

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
        update_fields = []
        args: dict = {'vacancy_id': vacancy_id}

        if name is not None:
            update_fields.append("name = :name")
            args['name'] = name

        if tags is not None:
            update_fields.append("tags = :tags")
            args['tags'] = tags

        if description is not None:
            update_fields.append("description = :description")
            args['description'] = description

        if red_flags is not None:
            update_fields.append("red_flags = :red_flags")
            args['red_flags'] = red_flags

        if skill_lvl is not None:
            update_fields.append("skill_lvl = :skill_lvl")
            args['skill_lvl'] = skill_lvl.value

        if not update_fields:
            return

        query = f"""
                UPDATE vacancies
                SET {', '.join(update_fields)}
                WHERE id = :vacancy_id;
                """

        await self.db.update(query, args)

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

        args = {
            'vacancy_id': vacancy_id,
            'question': question,
            'hint_for_evaluation': hint_for_evaluation,
            'weight': weight,
            'question_type': question_type.value,
            'response_time': response_time,
        }
        question_id = await self.db.insert(add_question_query, args)

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
    ) -> None:
        update_fields = []
        args: dict = {'question_id': question_id}

        if question is not None:
            update_fields.append("question = :question")
            args['question'] = question

        if hint_for_evaluation is not None:
            update_fields.append("hint_for_evaluation = :hint_for_evaluation")
            args['hint_for_evaluation'] = hint_for_evaluation

        if weight is not None:
            update_fields.append("weight = :weight")
            args['weight'] = weight

        if question_type is not None:
            update_fields.append("question_type = :question_type")
            args['question_type'] = question_type.value

        if response_time is not None:
            update_fields.append("response_time = :response_time")
            args['response_time'] = response_time

        if not update_fields:
            return

        query = f"""
                UPDATE vacancy_questions
                SET {', '.join(update_fields)}
                WHERE id = :question_id;
                """

        await self.db.update(query, args)

    @traced_method()
    async def delete_question(self, question_id: int) -> None:
        args = {'question_id': question_id}
        await self.db.delete(delete_question_query, args)

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
        args = {
            'vacancy_id': vacancy_id,
            'logic_structure_score_weight': logic_structure_score_weight,
            'soft_skill_score_weight': soft_skill_score_weight,
            'hard_skill_score_weight': hard_skill_score_weight,
            'accordance_xp_resume_score_weight': accordance_xp_resume_score_weight,
            'accordance_skill_resume_score_weight': accordance_skill_resume_score_weight,
            'red_flag_score_weight': red_flag_score_weight,
        }
        await self.db.insert(create_interview_weights_query, args)

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
        update_fields = []
        args = {'vacancy_id': vacancy_id}

        if logic_structure_score_weight is not None:
            update_fields.append("logic_structure_score_weight = :logic_structure_score_weight")
            args['logic_structure_score_weight'] = logic_structure_score_weight

        if soft_skill_score_weight is not None:
            update_fields.append("soft_skill_score_weight = :soft_skill_score_weight")
            args['soft_skill_score_weight'] = soft_skill_score_weight

        if hard_skill_score_weight is not None:
            update_fields.append("hard_skill_score_weight = :hard_skill_score_weight")
            args['hard_skill_score_weight'] = hard_skill_score_weight

        if accordance_xp_resume_score_weight is not None:
            update_fields.append("accordance_xp_resume_score_weight = :accordance_xp_resume_score_weight")
            args['accordance_xp_resume_score_weight'] = accordance_xp_resume_score_weight

        if accordance_skill_resume_score_weight is not None:
            update_fields.append("accordance_skill_resume_score_weight = :accordance_skill_resume_score_weight")
            args['accordance_skill_resume_score_weight'] = accordance_skill_resume_score_weight

        if red_flag_score_weight is not None:
            update_fields.append("red_flag_score_weight = :red_flag_score_weight")
            args['red_flag_score_weight'] = red_flag_score_weight

        if not update_fields:
            return

        query = f"""
                UPDATE interview_weights
                SET {', '.join(update_fields)}
                WHERE vacancy_id = :vacancy_id;
                """

        await self.db.update(query, args)

    @traced_method()
    async def create_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int,
            accordance_skill_vacancy_score_threshold: int,
            recommendation_weight: int,
            portfolio_weight: int,
    ) -> None:
        args = {
            'vacancy_id': vacancy_id,
            'accordance_xp_vacancy_score_threshold': accordance_xp_vacancy_score_threshold,
            'accordance_skill_vacancy_score_threshold': accordance_skill_vacancy_score_threshold,
            'recommendation_weight': recommendation_weight,
            'portfolio_weight': portfolio_weight,
        }
        await self.db.insert(create_resume_weights_query, args)

    @traced_method()
    async def edit_resume_weights(
            self,
            vacancy_id: int,
            accordance_xp_vacancy_score_threshold: int | None,
            accordance_skill_vacancy_score_threshold: int | None,
            recommendation_weight: int | None,
            portfolio_weight: int | None,
    ) -> None:
        update_fields = []
        args = {'vacancy_id': vacancy_id}

        if accordance_xp_vacancy_score_threshold is not None:
            update_fields.append("accordance_xp_vacancy_score_threshold = :accordance_xp_vacancy_score_threshold")
            args['accordance_xp_vacancy_score_threshold'] = accordance_xp_vacancy_score_threshold

        if accordance_skill_vacancy_score_threshold is not None:
            update_fields.append("accordance_skill_vacancy_score_threshold = :accordance_skill_vacancy_score_threshold")
            args['accordance_skill_vacancy_score_threshold'] = accordance_skill_vacancy_score_threshold

        if recommendation_weight is not None:
            update_fields.append("recommendation_weight = :recommendation_weight")
            args['recommendation_weight'] = recommendation_weight

        if portfolio_weight is not None:
            update_fields.append("portfolio_weight = :portfolio_weight")
            args['portfolio_weight'] = portfolio_weight

        if not update_fields:
            return

        query = f"""
                UPDATE resume_weights
                SET {', '.join(update_fields)}
                WHERE vacancy_id = :vacancy_id;
                """

        await self.db.update(query, args)

    @traced_method()
    async def get_vacancy_by_id(self, vacancy_id: int) -> list[model.Vacancy]:
        args = {'vacancy_id': vacancy_id}
        rows = await self.db.select(get_vacancy_by_id_query, args)
        vacancy = model.Vacancy.serialize(rows) if rows else []

        return vacancy

    @traced_method()
    async def get_all_vacancy(self) -> list[model.Vacancy]:
        rows = await self.db.select(get_all_vacancy_query, {})
        vacancies = model.Vacancy.serialize(rows) if rows else []

        return vacancies

    @traced_method()
    async def get_all_question(self, vacancy_id: int) -> list[model.VacancyQuestion]:
        args = {'vacancy_id': vacancy_id}
        rows = await self.db.select(get_all_question_query, args)
        questions = model.VacancyQuestion.serialize(rows) if rows else []

        return questions

    @traced_method()
    async def get_interview_weights(self, vacancy_id: int) -> list[model.InterviewWeights]:
        args = {'vacancy_id': vacancy_id}
        rows = await self.db.select(get_interview_weights_query, args)
        weights = model.InterviewWeights.serialize(rows) if rows else []

        return weights

    @traced_method()
    async def get_question_by_id(self, question_id: int) -> list[model.VacancyQuestion]:

        args = {'question_id': question_id}
        rows = await self.db.select(get_question_by_id_query, args)
        questions = model.VacancyQuestion.serialize(rows) if rows else []

        return questions

    @traced_method()
    async def get_resume_weights(self, vacancy_id: int) -> list[model.ResumeWeights]:
        args = {'vacancy_id': vacancy_id}
        rows = await self.db.select(get_resume_weights_query, args)
        weights = model.ResumeWeights.serialize(rows) if rows else []

        return weights
