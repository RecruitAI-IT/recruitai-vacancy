from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class SkillLevel(Enum):
    JUNIOR = "junior"
    MIDDLE = "middle"
    SENIOR = "senior"
    LEAD = "lead"


class QuestionsType(Enum):
    SOFT = "soft"
    HARD = "hard"
    SOFT_HARD = "soft-hard"


@dataclass
class Vacancy:
    id: int
    name: str
    tags: list[str]
    description: str
    red_flags: str
    skill_lvl: SkillLevel

    created_at: datetime

    @classmethod
    def serialize(cls, rows) -> List['Vacancy']:
        return [
            cls(
                id=row.id,
                name=row.name,
                tags=row.tags,
                description=row.description,
                red_flags=row.red_flags,
                skill_lvl=SkillLevel(row.skill_lvl),
                created_at=row.created_at
            )
            for row in rows
        ]

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "tags": self.tags,
            "description": self.description,
            "red_flags": self.red_flags,
            "skill_lvl": self.skill_lvl.value
        }


@dataclass
class VacancyQuestion:
    id: int
    vacancy_id: int
    question: str
    hint_for_evaluation: str
    weight: int
    question_type: QuestionsType
    response_time: int

    created_at: datetime

    @classmethod
    def serialize(cls, rows) -> List['VacancyQuestion']:
        return [
            cls(
                id=row.id,
                vacancy_id=row.vacancy_id,
                question=row.question,
                hint_for_evaluation=row.hint_for_evaluation,
                weight=row.weight,
                question_type=QuestionsType(row.question_type),
                response_time=row.response_time,
                created_at=row.created_at
            )
            for row in rows
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vacancy_id": self.vacancy_id,
            "question": self.question,
            "hint_for_evaluation": self.hint_for_evaluation,
            "weight": self.weight,
            "question_type": self.question_type.value,
            "response_time": self.response_time
        }


@dataclass
class InterviewWeights:
    id: int
    vacancy_id: int
    logic_structure_score_weight: int
    soft_skill_score_weight: int
    hard_skill_score_weight: int
    accordance_xp_resume_score_weight: int
    accordance_skill_resume_score_weight: int
    red_flag_score_weight: int

    created_at: datetime

    @classmethod
    def serialize(cls, rows) -> List['InterviewWeights']:
        return [
            cls(
                id=row.id,
                vacancy_id=row.vacancy_id,
                logic_structure_score_weight=row.logic_structure_score_weight,
                soft_skill_score_weight=row.soft_skill_score_weight,
                hard_skill_score_weight=row.hard_skill_score_weight,
                accordance_xp_resume_score_weight=row.accordance_xp_resume_score_weight,
                accordance_skill_resume_score_weight=row.accordance_skill_resume_score_weight,
                red_flag_score_weight=row.red_flag_score_weight,
                created_at=row.created_at
            )
            for row in rows
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vacancy_id": self.vacancy_id,
            "logic_structure_score_weight": self.logic_structure_score_weight,
            "soft_skill_score_weight": self.soft_skill_score_weight,
            "hard_skill_score_weight": self.hard_skill_score_weight,
            "accordance_xp_resume_score_weight": self.accordance_xp_resume_score_weight,
            "accordance_skill_resume_score_weight": self.accordance_skill_resume_score_weight,
            "red_flag_score_weight": self.red_flag_score_weight
        }


@dataclass
class ResumeWeights:
    id: int
    vacancy_id: int
    accordance_xp_vacancy_score_threshold: int
    accordance_skill_vacancy_score_threshold: int
    recommendation_weight: int
    portfolio_weight: int

    created_at: datetime

    @classmethod
    def serialize(cls, rows) -> List['ResumeWeights']:
        return [
            cls(
                id=row.id,
                vacancy_id=row.vacancy_id,
                accordance_xp_vacancy_score_threshold=row.accordance_xp_vacancy_score_threshold,
                accordance_skill_vacancy_score_threshold=row.accordance_skill_vacancy_score_threshold,
                recommendation_weight=row.recommendation_weight,
                portfolio_weight=row.portfolio_weight,
                created_at=row.created_at
            )
            for row in rows
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vacancy_id": self.vacancy_id,
            "accordance_xp_vacancy_score_threshold": self.accordance_xp_vacancy_score_threshold,
            "accordance_skill_vacancy_score_threshold": self.accordance_skill_vacancy_score_threshold,
            "recommendation_weight": self.recommendation_weight,
            "portfolio_weight": self.portfolio_weight
        }
