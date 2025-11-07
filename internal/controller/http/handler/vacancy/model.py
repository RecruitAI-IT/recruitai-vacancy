from pydantic import BaseModel
from internal import model


class CreateVacancyBody(BaseModel):
    name: str
    tags: list[str]
    description: str
    red_flags: str
    skill_lvl: model.SkillLevel


class EditVacancyBody(BaseModel):
    vacancy_id: int
    name: str | None
    tags: list[str] | None
    description: str | None
    red_flags: str | None
    skill_lvl: model.SkillLevel | None


class AddQuestionBody(BaseModel):
    vacancy_id: int
    question: str
    hint_for_evaluation: str
    weight: int
    question_type: model.QuestionsType
    response_time: int


class EditQuestionBody(BaseModel):
    question_id: int
    vacancy_id: int
    question: str | None
    hint_for_evaluation: str | None
    weight: int | None  # [0;10]
    question_type: model.QuestionsType | None
    response_time: int | None

class CreateInterviewWeightsBody(BaseModel):
    vacancy_id: int
    logic_structure_score_weight: int
    soft_skill_score_weight: int
    hard_skill_score_weight: int
    accordance_xp_resume_score_weight: int
    accordance_skill_resume_score_weight: int
    red_flag_score_weight: int


class EditInterviewWeightsBody(BaseModel):
    vacancy_id: int
    logic_structure_score_weight: int | None
    soft_skill_score_weight: int | None
    hard_skill_score_weight: int | None
    accordance_xp_resume_score_weight: int | None
    accordance_skill_resume_score_weight: int | None
    red_flag_score_weight: int | None


class CreateResumeWeightsBody(BaseModel):
    vacancy_id: int
    accordance_xp_vacancy_score_threshold: int
    accordance_skill_vacancy_score_threshold: int
    recommendation_weight: int
    portfolio_weight: int

class EditResumeWeightsBody(BaseModel):
    vacancy_id: int
    accordance_xp_vacancy_score_threshold: int | None
    accordance_skill_vacancy_score_threshold: int | None
    recommendation_weight: int | None
    portfolio_weight: int | None

class GenerateQuestionBody(BaseModel):
    vacancy_id: int
    questions_type: model.QuestionsType
    count_questions: int


class GenerateQuestionResponse(BaseModel):
    questions: list[model.VacancyQuestion]

class GenerateTagsBody(BaseModel):
    vacancy_description: str

class GenerateTagsResponse(BaseModel):
    tags: list[str]

class EvaluateResumeResponse(BaseModel):
    class EvaluationResume(BaseModel):
        candidate_email: str
        candidate_name: str
        candidate_phone: str
        accordance_xp_vacancy_score: int
        accordance_skill_vacancy_score: int

    evaluation_resumes: list[EvaluationResume]

class RespondResponse(BaseModel):
    interview_link: str
    accordance_xp_vacancy_score: int
    accordance_skill_vacancy_score: int
    message_to_candidate: str




