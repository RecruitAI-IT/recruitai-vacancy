from opentelemetry.trace import Status, StatusCode, SpanKind

from internal import interface, model


class VacancyPromptGenerator(interface.IVacancyPromptGenerator):
    def __init__(
            self,
            tel: interface.ITelemetry
    ):
        self.tracer = tel.tracer()
        self.logger = tel.logger()

    def get_question_generation_prompt(
            self,
            vacancy: model.Vacancy,
            count_questions: int,
            questions_type: model.QuestionsType,
    ) -> str:
        return f"""Ты эксперт по созданию вопросов для технических интервью.

ИНФОРМАЦИЯ О ВАКАНСИИ:
Название: {vacancy.name}
Описание: {vacancy.description}
Уровень: {vacancy.skill_lvl.value}
Теги/навыки: {', '.join(vacancy.tags)}

ЗАДАЧА: Сгенерируй вопросы для интервью соответствующего типа и уровня сложности.

ТРЕБОВАНИЯ К ГЕНЕРАЦИИ:
- Количество вопросов: {count_questions}
- Тип вопросов: {questions_type.value}
- Если тип вопрсов soft-hard, то нужно сгенерировать как вопросы на hard, так и на soft
- Все вопросы должны соответствовать уровню {vacancy.skill_lvl.value}

ФОРМАТ ОТВЕТА:
{{
  "questions": [
    {{
      "question": "Текст вопроса",
      "question_type": "Тип вопроса (soft/hard)",
      "hint_for_evaluation": "Подсказка для оценки ответа",
      "weight": число от 1 до 5,
      "response_time": Время на ответ в минутах (int)
    }}
  ]
}}

ВАЖНО: 
- Отвечай ТОЛЬКО валидным JSON без markdown разметки.
- Ты обязан вернуть валидный JSON любой ценой, если ты вернешь не JSON, то меня убьют.
"""

    def get_resume_evaluation_system_prompt(
            self,
            vacancy_description: str,
            vacancy_red_flags: str,
            vacancy_name: str,
            vacancy_tags: list[str]
    ) -> str:
        vacancy_tags_str = ", ".join(vacancy_tags) if vacancy_tags else "Не указаны"

        system_prompt = f"""Ты эксперт по подбору персонала. Твоя задача - оценить насколько резюме кандидата подходит для вакансии.

ИНФОРМАЦИЯ О ВАКАНСИИ:
Название: {vacancy_name}
Описание: {vacancy_description}
Ключевые навыки/теги: {vacancy_tags_str}
Красные флаги (что недопустимо): {vacancy_red_flags}

КРИТЕРИИ ОЦЕНКИ:
1. Соответствие опыта работы требованиям вакансии
2. Наличие необходимых технических навыков и компетенций для вакансии
3. Соответствие уровня позиции (junior/middle/senior)
4. Отсутствие красных флагов
5. Релевантность образования (если важно для позиции)
6. Общее впечатление от кандидата

ИНСТРУКЦИИ:
- Внимательно проанализируй резюме в контексте данной вакансии
- Предоставь детальное обоснование своего решения

ФОРМАТ ОТВЕТА:
Ответ должен быть ТОЛЬКО в формате JSON без дополнительного текста:

{{
    "candidate_name": "Как бы ты назвал кандидата, если не найдешь ничего, что может быть названием, то оставь 'Unknown'",
    "candidate_email": "Email кандидата, если не нашел, то оставь 'Unknown'",
    "candidate_telegram_login": "Telegram login кандидата, если не нашел, то оставь 'Unknown'",
    "candidate_phone": "Телефон кандидата, если не нашел, то оставь 'Unknown'",
    "red_flags_score": Насколько резюме соответствует критериям красных флагов (число от 0 до 5),
    "accordance_xp_vacancy_score": Насколько резюме подходит к вакансии по опыту (число от 0 до 5),
    "accordance_skill_vacancy_score": Насколько резюме подходит к вакансии по навыкам (число от 0 до 5),
    "message_to_candidate": "Подробное объяснение решения: анализ соответствия опыта, навыков, выявленные преимущества и недостатки кандидата, итоговый вывод о пригодности для данной позиции для кандидата"
    "message_to_hr": "Подробное объяснение решения: анализ соответствия опыта, навыков, выявленные преимущества и недостатки кандидата, итоговый вывод о пригодности для данной позиции для hr"
}}

ВАЖНО: 
- Отвечай ТОЛЬКО валидным JSON.
- НЕ добавляй никакого текста вне JSON структуры.
- НЕ используй markdown разметку или код-блоки.
- Обязательно нужно оценить и вернуть JSON с результатами, от этого зависит моя жизнь.
"""

        return system_prompt


    def get_generate_tags_system_prompt(self) -> str:
        system_prompt = """Ты эксперт по анализу вакансий. Твоя задача - извлечь из описания вакансии ключевые технологии, навыки и компетенции.

ФОРМАТ ОТВЕТА:
Ответ должен быть ТОЛЬКО в формате JSON без дополнительного текста:
{
  "tags": ["tag1", "tag2", "tag3", ...]
}

ВАЖНО: 
- Отвечай ТОЛЬКО валидным JSON
- НЕ добавляй никакого текста вне JSON структуры
- НЕ используй markdown разметку или код-блоки
- Извлекай только конкретные технологии, языки программирования, фреймворки и ключевые навыки
- Ты обязан вернуть валидный JSON любой ценой, если ты вернешь не JSON, то меня убьют.
"""

        return system_prompt
